#!/usr/bin/env python3
"""
Evaluate VLMs on PhysBench single-image records.

Usage:
    python evaluate_physbench.py --model Qwen/Qwen3-VL-8B-Instruct \
        --backend vllm --server_url http://localhost:8010
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image

from extract_answer import (
    strip_thinking, extract_letter_from_response,
    extract_with_confirmation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHYSBENCH_ROOT_DEFAULT = Path(__file__).resolve().parent.parent.parent / "dataset" / "PhysBench"

SYSTEM_PROMPT = (
    "You are an expert visual reasoning assistant evaluating physical AI scenarios."
)

MAX_RETRIES = 2
MAX_RETRY_TOKENS = 4096
EXTRA_BODY = None
ALLOW_THINKING = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def query_vlm(client: OpenAI, model: str, image_uri: str,
              question: str, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": question},
        ]},
    ]
    cur_tokens = max_tokens
    for _ in range(MAX_RETRIES):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=cur_tokens, temperature=0.0,
            **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
        )
        content = resp.choices[0].message.content
        if content is not None:
            return content.strip()
        cur_tokens = min(cur_tokens * 2, MAX_RETRY_TOKENS)
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=512, temperature=0.0)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_physbench(physbench_root: Path):
    """Load single-image records from PhysBench (original or verified format)."""
    manifest_path = physbench_root / "manifest.json"
    test_path = physbench_root / "test.json"

    if manifest_path.exists() and not test_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return [
            {"idx": r["doc_id"], "question": r["question"],
             "image_path": str(physbench_root / r["frame_path"]),
             "correct_letter": r.get("answer", ""),
             "task_type": r.get("category", ""),
             "sub_type": r.get("subcategory", ""),
             "ability_type": r.get("ability_type", "")}
            for r in manifest.get("records", [])
            if (physbench_root / r["frame_path"]).exists()
            and r.get("answer") in ("A", "B", "C", "D")
        ]

    test_data = json.loads(test_path.read_text())
    answers = {r['idx']: r for r in json.loads(
        (physbench_root / "test_answer.json").read_text())}
    records = []
    for r in test_data:
        imgs = [f for f in r['file_name'] if f.endswith(('.jpg', '.png', '.jpeg'))]
        vids = [f for f in r['file_name'] if f.endswith('.mp4')]
        if len(imgs) != 1 or vids:
            continue
        img_path = physbench_root / "image" / imgs[0]
        ans = answers.get(r['idx'], {})
        cl = ans.get('answer', '')
        if not img_path.exists() or cl not in ('A', 'B', 'C', 'D'):
            continue
        records.append({
            "idx": r['idx'], "question": r['question'],
            "image_path": str(img_path), "correct_letter": cl,
            "task_type": ans.get('task_type', ''),
            "sub_type": ans.get('sub_type', ''),
            "ability_type": ans.get('ability_type', ''),
        })
    return records


# ---------------------------------------------------------------------------
# Query + extraction
# ---------------------------------------------------------------------------

def process_record(client, model, record, max_tokens, extractor=None):
    question = record['question']
    if not ALLOW_THINKING:
        question += "\nBriefly explain your reasoning. Put your final answer (a single letter) on the last line."

    try:
        image_uri = encode_image(Path(record['image_path']))
        response = query_vlm(client, model, image_uri, question, max_tokens)

        # Quick regex → LLM extractor
        chosen = extract_letter_from_response(response)
        if chosen is None and extractor:
            ext_client, ext_model = extractor
            chosen = extract_with_confirmation(
                response, record['question'],
                extractor_client=ext_client, extractor_model=ext_model)

        return {
            "idx": record['idx'], "question": record['question'],
            "response": response, "chosen_letter": chosen,
            "correct_letter": record['correct_letter'],
            "is_correct": chosen == record['correct_letter'],
            "task_type": record['task_type'],
            "sub_type": record['sub_type'],
            "ability_type": record['ability_type'],
            "status": "ok",
        }
    except Exception as e:
        return {
            "idx": record['idx'], "question": record['question'],
            "response": str(e), "chosen_letter": None,
            "correct_letter": record['correct_letter'],
            "is_correct": False, "task_type": record['task_type'],
            "sub_type": record['sub_type'],
            "ability_type": record['ability_type'],
            "status": "error",
        }


# ---------------------------------------------------------------------------
# Metrics & output
# ---------------------------------------------------------------------------

def compute_metrics(results):
    ok = [r for r in results if r['status'] == 'ok']
    total, correct = len(ok), sum(1 for r in ok if r['is_correct'])
    buckets = {"task": defaultdict(lambda: [0, 0]),
               "sub": defaultdict(lambda: [0, 0]),
               "ability": defaultdict(lambda: [0, 0])}
    for r in ok:
        for key, btype in [(r['task_type'], 'task'),
                           (r['sub_type'], 'sub'),
                           (r['ability_type'], 'ability')]:
            buckets[btype][key][1] += 1
            if r['is_correct']:
                buckets[btype][key][0] += 1

    def acc(b):
        return {k: v[0]/v[1] if v[1] else 0.0 for k, v in sorted(b.items())}

    return {
        "overall": correct / total if total else 0.0,
        "n_total": total, "n_correct": correct,
        "by_task_type": acc(buckets["task"]),
        "by_sub_type": acc(buckets["sub"]),
        "by_ability_type": acc(buckets["ability"]),
    }


def _save_checkpoint(output_dir, model, backend, results, total):
    """Save intermediate summary checkpoint."""
    metrics = compute_metrics(results)
    with open(output_dir / "summary_checkpoint.json", "w") as f:
        json.dump({"model": model, "backend": backend,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "progress": f"{len(results)}/{total}",
                   "n_done": len(results), "n_total": total,
                   "metrics": metrics}, f, indent=2)


def print_summary(metrics, model):
    print(f"\n{'='*60}\n  {model}")
    print(f"  Overall: {metrics['overall']*100:.1f}% "
          f"({metrics['n_correct']}/{metrics['n_total']})\n{'='*60}")
    for label, key in [("Task type", "by_task_type"),
                       ("Sub type", "by_sub_type"),
                       ("Ability", "by_ability_type")]:
        print(f"\n  {label}:")
        for k, v in metrics[key].items():
            print(f"    {k:20s}: {v*100:.1f}%")
    print()


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLMs on PhysBench")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["vllm", "openrouter"], default="vllm")
    p.add_argument("--server_url", default="http://localhost:8010")
    p.add_argument("--api_key", default="dummy")
    p.add_argument("--physbench_root", default=str(PHYSBENCH_ROOT_DEFAULT))
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--extractor_backend", choices=["vllm", "openrouter"],
                   default="openrouter")
    p.add_argument("--extractor_model", default="qwen/qwen3-8b")
    p.add_argument("--extractor_url", default=None)
    p.add_argument("--task_types", nargs="+", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global EXTRA_BODY, ALLOW_THINKING

    # Client
    if args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", args.api_key)
        if api_key == "dummy":
            sys.exit("ERROR: set OPENROUTER_API_KEY")
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=api_key, timeout=300.0)
    else:
        try:
            requests.get(f"{args.server_url}/v1/models", timeout=5).raise_for_status()
        except Exception as e:
            sys.exit(f"ERROR: server not reachable: {e}")
        client = OpenAI(base_url=f"{args.server_url}/v1",
                        api_key=args.api_key, timeout=300.0)
    print(f"Backend: {args.backend}  model={args.model}")

    if args.enable_thinking:
        ALLOW_THINKING = True
        print("  Thinking: ENABLED")
    else:
        EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
        print("  Thinking: DISABLED (default)")

    # Data
    records = load_physbench(Path(args.physbench_root))
    print(f"Loaded {len(records)} single-image records")
    if args.task_types:
        records = [r for r in records if r['task_type'] in args.task_types]
    if args.max_samples:
        records = records[:args.max_samples]

    # Extractor
    if args.extractor_backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "dummy")
        if api_key == "dummy":
            sys.exit("ERROR: set OPENROUTER_API_KEY for extractor")
        ext_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                            api_key=api_key, timeout=60.0)
    else:
        if not args.extractor_url:
            sys.exit("ERROR: --extractor_url required")
        ext_client = OpenAI(base_url=f"{args.extractor_url}/v1",
                            api_key="dummy", timeout=60.0)
    extractor = (ext_client, args.extractor_model)

    # Output
    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("eval_results") / "physbench" / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run with incremental saving
    print(f"\nEvaluating {len(records)} queries...")
    t0 = time.time()
    results = []
    jsonl_path = output_dir / "results.jsonl"
    save_every = 50  # save checkpoint every N results

    # Open jsonl for incremental writes
    jsonl_f = open(jsonl_path, "w")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, client, args.model, rec,
                               args.max_tokens, extractor): rec
                   for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            # Write each result immediately
            jsonl_f.write(json.dumps(res) + "\n")
            jsonl_f.flush()
            m = "+" if res["is_correct"] else ("-" if res["status"] == "ok" else "E")
            print(f"\r  {i}/{len(records)} [{m}] {res['task_type']}/{res['idx']}",
                  end="", flush=True)
            # Save checkpoint summary periodically
            if i % save_every == 0:
                _save_checkpoint(output_dir, args.model, args.backend, results, len(records))

    jsonl_f.close()
    print(f"\n  Done ({time.time()-t0:.1f}s)")

    # Final save
    metrics = compute_metrics(results)
    print_summary(metrics, args.model)

    # Overwrite jsonl sorted
    results.sort(key=lambda r: r['idx'])
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(output_dir / "physbench_submission.json", "w") as f:
        json.dump([{"idx": r["idx"], "answer": r["chosen_letter"] or ""}
                   for r in sorted(results, key=lambda r: r['idx'])], f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"model": args.model, "backend": args.backend,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "n_records": len(records), "metrics": metrics}, f, indent=2)
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
