#!/usr/bin/env python3
"""
Verify target images for visual adversarial attacks.

Sends each target image + question to a vllm server and checks whether the
model already answers with attack_target_text -- confirming the target image
is a good attack reference (works for FOA, M-Attack, and any target-image attack).

A "pass" means the target image causes the VLM to give the wrong (target) answer,
which is what the adversarial attack aims to achieve on the source image.

Usage:
    python -m agent.target_generation.verify_targets \
        --dataset_dir dataset/redteam \
        --server http://localhost:8001 \
        --model Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image


CATEGORIES = [
    "object", "exist_yes_to_no", "exist_no_to_yes",
    "prediction", "perception", "status", "count",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify FOA target images via vllm server")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory containing per-category manifest.json files")
    parser.add_argument("--server", type=str, default="http://localhost:8001",
                        help="vllm server base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--api_key", type=str, default="dummy")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Subset of categories to verify (default: all)")
    return parser.parse_args()


def check_server(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def query_vllm(client: OpenAI, model: str, image_path: Path, question: str) -> str:
    img_data = encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_data}},
                {"type": "text", "text": question},
            ],
        }],
        max_tokens=64,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def find_mcq_letter(question: str, target_text: str) -> str:
    """For MCQ questions, find the option letter matching the target text."""
    # Match patterns like "A. Turn left." "B. Going ahead."
    for match in re.finditer(r'([A-C])\.\s*([^.]+\.)', question):
        letter, text = match.group(1), match.group(2).strip()
        if text.lower() == target_text.lower():
            return letter
    return target_text  # fallback: match text directly


def is_match(response: str, attack_target: str, extractor: str, question: str) -> bool:
    resp = response.lower().strip()
    target = attack_target.lower().strip()

    if extractor == "yes_no":
        # Accept if response starts with the target word
        return resp.startswith(target) or resp == target

    elif extractor == "mcq":
        # Check both the letter and the text
        letter = find_mcq_letter(question, attack_target)
        return (resp.startswith(letter.lower()) or
                resp.startswith(target) or
                target in resp)

    elif extractor == "count":
        word_map = {"0": ["0", "zero", "none", "no "], "1": ["1", "one"],
                    "2": ["2", "two"], "3": ["3", "three"]}
        candidates = word_map.get(target, [target])
        return any(resp.startswith(c) or f" {c}" in resp for c in candidates)

    else:  # open
        # Semantic synonyms for motion-state targets
        SYNONYMS = {
            "moving":  ["moving", "in motion", "driving", "traveling"],
            "driving": ["driving", "moving", "in motion", "traveling"],
            "stopped": ["stopped", "stationary", "not moving", "parked"],
        }
        candidates = SYNONYMS.get(target, [target])
        return any(c in resp for c in candidates)


def main():
    args = parse_args()
    foa_dir = Path(args.dataset_dir)
    categories = args.categories or CATEGORIES

    # Check server health
    if not check_server(args.server):
        print(f"ERROR: vllm server not reachable at {args.server}/health")
        sys.exit(1)
    print(f"Server OK: {args.server}  model={args.model}\n")

    client = OpenAI(base_url=f"{args.server}/v1", api_key=args.api_key)

    # Load answer_extractor mapping from behaviors.jsonl (optional, fall back to per-category)
    CAT_EXTRACTOR = {
        "object": "open",
        "exist_yes_to_no": "yes_no",
        "exist_no_to_yes": "yes_no",
        "prediction": "yes_no",
        "perception": "mcq",
        "status": "open",
        "count": "count",
    }

    overall_pass = 0
    overall_total = 0
    category_results = {}

    for category in categories:
        manifest_path = foa_dir / category / "manifest.json"
        if not manifest_path.exists():
            print(f"[{category}] manifest.json not found, skipping")
            continue

        manifest = json.load(open(manifest_path))
        extractor = CAT_EXTRACTOR[category]
        records = manifest["records"]

        print(f"[{category.upper()}]  extractor={extractor}")
        cat_pass = 0
        results = []

        for rec in records:
            tgt_path = Path(rec["target_path"])
            if not tgt_path.exists():
                status = "MISSING"
                response = ""
                print(f"  {rec['index']}: MISSING target image")
            else:
                try:
                    response = query_vllm(client, args.model, tgt_path, rec["question"])
                    passed = is_match(response, rec["attack_target_text"], extractor, rec["question"])
                    status = "PASS" if passed else "FAIL"
                    if passed:
                        cat_pass += 1
                    marker = "PASS" if passed else "FAIL"
                    print(f"  {rec['index']}: {marker}  target={rec['attack_target_text']!r:20s}  "
                          f"response={response[:60]!r}")
                except Exception as e:
                    status = "ERROR"
                    response = str(e)
                    print(f"  {rec['index']}: ERROR  {e}")

            results.append({**rec, "response": response, "status": status})

        total = len([r for r in results if r["status"] != "MISSING"])
        print(f"  -> {cat_pass}/{total} pass ({100*cat_pass/total:.0f}%)\n" if total else "  -> 0/0\n")
        overall_pass += cat_pass
        overall_total += total
        category_results[category] = {"pass": cat_pass, "total": total, "records": results}

    # Save results
    out_path = foa_dir / "verification_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "overall_pass": overall_pass,
            "overall_total": overall_total,
            "categories": {
                cat: {"pass": v["pass"], "total": v["total"]}
                for cat, v in category_results.items()
            },
            "details": category_results,
        }, f, indent=2)

    print("=" * 60)
    print(f"Overall: {overall_pass}/{overall_total} "
          f"({100*overall_pass/overall_total:.0f}%)" if overall_total else "Overall: 0/0")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
