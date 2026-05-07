#!/usr/bin/env python3
"""
Attack-scenario agent: build target images for PAI-bench and PhysBench.

Reads from behaviors.jsonl (PAI-bench) or manifest.json (PhysBench),
uses local vLLM servers to reason about each scenario, generate target images,
and verify them against a victim VLM.

Usage:
    # PAI-bench (from behaviors.jsonl)
    python -m agent.target_generation.build_targets \
        --data_root       dataset/pai_bench_full_dataset_v2 \
        --domains pai-bench \
        --thinking_server http://localhost:8000 \
        --gen_server      http://localhost:8091 \
        --vllm_server     http://localhost:8001 \
        --max_attempts 3 --workers 3

    # PhysBench with editing for annotated questions
    python -m agent.target_generation.build_targets \
        --data_root       dataset/physbench-verified \
        --domains physbench \
        --thinking_server http://localhost:8000 \
        --gen_server      http://localhost:8091 \
        --edit_server     http://localhost:8092 \
        --vllm_server     http://localhost:8001 \
        --max_attempts 3 --workers 3
"""

import argparse
import json
import os
import sys
import threading
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PAIBENCH_SUBCAT_DIR,
    normalize_paibench_record,
    normalize_physbench_record,
)
from agent.target_generation.agent_loop import AgentConfig, run_agent_for_record


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def load_records(
    data_root: Path,
    domains: list[str],
    categories: list[str] | None,
    limit: int | None,
) -> list[dict]:
    """Load and normalize records from behaviors.jsonl and/or physbench manifest."""
    records = []
    scenario_counts: dict[str, int] = {}

    # Load PAI-bench from behaviors.jsonl or manifest.json
    if "pai-bench" in domains:
        behaviors_path = data_root / "behaviors.jsonl"
        manifest_path = data_root / "manifest.json"

        raw_records = []
        if behaviors_path.exists():
            with open(behaviors_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    if raw.get("domain", "") == "pai-bench":
                        raw_records.append(raw)
            print(f"  Loaded {len(raw_records)} PAI-bench records from behaviors.jsonl")
        elif manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            raw_records = data.get("records", [])
            print(f"  Loaded {len(raw_records)} PAI-bench records from manifest.json")
        else:
            print(f"ERROR: neither {behaviors_path} nor {manifest_path} found",
                  file=sys.stderr)
            if "physbench" not in domains:
                sys.exit(1)

        for raw in raw_records:
            rec = normalize_paibench_record(raw, data_root)
            if rec is None:
                continue

            cat = rec["category"]
            if categories and cat not in categories:
                continue

            if limit is not None:
                scenario_key = PAIBENCH_SUBCAT_DIR.get(cat, cat)
                count = scenario_counts.get(scenario_key, 0)
                if count >= limit:
                    continue
                scenario_counts[scenario_key] = count + 1

            records.append(rec)

    # Load PhysBench from manifest.json
    if "physbench" in domains:
        manifest_path = data_root / "manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: {manifest_path} not found", file=sys.stderr)
            sys.exit(1)

        data = json.loads(manifest_path.read_text())
        for raw in data.get("records", []):
            rec = normalize_physbench_record(raw, data_root)
            if rec is None:
                continue

            cat = rec["category"]
            if categories and cat not in categories:
                continue

            if limit is not None:
                scenario_key = f"{cat}/{rec.get('subcategory', '')}"
                count = scenario_counts.get(scenario_key, 0)
                if count >= limit:
                    continue
                scenario_counts[scenario_key] = count + 1

            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Attack-scenario agent: build target images for "
                    "PAI-bench and PhysBench domains."
    )
    p.add_argument("--data_root", required=True,
                   help="Root data directory (behaviors.jsonl for PAI-bench, "
                        "manifest.json for PhysBench)")
    p.add_argument("--target_store", default=None,
                   help="Canonical target image store (default: {data_root}/target_images)")
    p.add_argument("--thinking_backend", choices=["vllm", "openrouter"],
                   default="openrouter",
                   help="Backend for thinking model (default: openrouter)")
    p.add_argument("--thinking_server", default=None,
                   help="vLLM server URL for thinking model (if backend=vllm)")
    p.add_argument("--thinking_model", default="qwen/qwen3-vl-32b-instruct",
                   help="Reasoning model for steps 1 and 4")
    p.add_argument("--gen_server", default="http://localhost:8091",
                   help="Local vllm-omni server for image generation")
    p.add_argument("--gen_model", default="Qwen/Qwen-Image-2512",
                   help="Image generation model for step 2")
    p.add_argument("--vllm_server", default="http://localhost:8001",
                   help="Local vLLM server for VLM verification")
    p.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="VLM model name on vllm server")
    p.add_argument("--max_attempts", type=int, default=3,
                   help="Max generate->verify->refine iterations per record")
    p.add_argument("--gen_sleep", type=int, default=0,
                   help="Seconds to sleep after each image generation call")
    p.add_argument("--workers", type=int, default=3,
                   help="Parallel workers")
    p.add_argument("--domains", nargs="+", default=["pai-bench"],
                   help="Domains to process: pai-bench, physbench")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Filter by category/subcategory names")
    p.add_argument("--limit", type=int, default=None,
                   help="First N records per category (for testing)")
    p.add_argument("--skip_source_verify", action="store_true",
                   help="Skip Step 0 source verification (generate targets for all records)")

    # Image editing (Qwen-Image-Edit-2511 via vLLM-Omni)
    p.add_argument("--edit_server", default=None,
                   help="vLLM-Omni server for image editing (e.g. http://localhost:8092)")
    p.add_argument("--edit_model", default="Qwen/Qwen-Image-Edit-2511",
                   help="Image editing model name on edit server")
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    target_store = Path(args.target_store) if args.target_store else data_root / "target_images"
    target_store.mkdir(parents=True, exist_ok=True)

    # -- Load records --
    records = load_records(data_root, args.domains, args.categories, args.limit)
    if not records:
        print("No records to process. Check --domains, --categories, and --data_root.")
        sys.exit(0)

    # -- Build API clients --
    # Thinking model: OpenRouter (default) or local vLLM
    if args.thinking_backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            sys.exit("ERROR: set OPENROUTER_API_KEY for thinking model")
        think_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=300.0,
        )
        print(f"Thinking: openrouter  ({args.thinking_model})")
    else:
        if not args.thinking_server:
            sys.exit("ERROR: --thinking_server required when --thinking_backend=vllm")
        think_client = OpenAI(
            base_url=f"{args.thinking_server}/v1",
            api_key="dummy",
            timeout=300.0,
        )
        print(f"Thinking: {args.thinking_server}  ({args.thinking_model})")
        try:
            r = requests.get(f"{args.thinking_server}/v1/models", timeout=5)
            print(f"  Thinking server {'OK' if r.status_code == 200 else f'returned {r.status_code}'}")
        except Exception as e:
            print(f"  WARNING: Thinking server not reachable ({e})")

    # Image generation: local vLLM-Omni
    gen_client = OpenAI(
        base_url=f"{args.gen_server}/v1",
        api_key="dummy",
        timeout=300.0,
    )
    print(f"Image gen: {args.gen_server}  ({args.gen_model})")
    try:
        r = requests.get(f"{args.gen_server}/v1/models", timeout=5)
        print(f"  Image gen server {'OK' if r.status_code == 200 else f'returned {r.status_code}'}")
    except Exception as e:
        print(f"  WARNING: Image gen server not reachable ({e})")

    # Optional: local vLLM for verification
    vlm_client = None
    try:
        r = requests.get(f"{args.vllm_server}/v1/models",
                         headers={"Authorization": "Bearer dummy"}, timeout=5)
        if r.status_code == 200:
            vlm_client = OpenAI(base_url=f"{args.vllm_server}/v1", api_key="dummy", timeout=300.0)
            print(f"VLM verify: {args.vllm_server}  ({args.vlm_model})")
        else:
            print(f"WARNING: VLM server returned {r.status_code} -- verification skipped")
    except Exception as e:
        print(f"WARNING: VLM server not reachable ({e}) -- verification skipped")

    # Optional: image editing server for annotated PhysBench
    edit_client = None
    if args.edit_server:
        try:
            r = requests.get(f"{args.edit_server}/v1/models", timeout=5)
            if r.status_code == 200:
                edit_client = OpenAI(
                    base_url=f"{args.edit_server}/v1",
                    api_key="dummy", timeout=300.0)
                print(f"Image edit: {args.edit_server}  ({args.edit_model})")
            else:
                print(f"WARNING: Edit server returned {r.status_code} -- edit mode disabled")
        except Exception as e:
            print(f"WARNING: Edit server not reachable ({e}) -- edit mode disabled")

    cfg = AgentConfig(
        think_client=think_client,
        think_model=args.thinking_model,
        gen_client=gen_client,
        gen_model=args.gen_model,
        vlm_client=vlm_client,
        vlm_model=args.vlm_model,
        max_attempts=args.max_attempts,
        gen_sleep=args.gen_sleep,
        skip_source_verify=args.skip_source_verify,
        edit_client=edit_client,
        edit_model=args.edit_model,
    )

    # -- Summary --
    domain_counts = Counter(r["domain"] for r in records)
    cat_counts = Counter(r["category"] for r in records)
    print(f"\nTotal records: {len(records)}")
    for d, n in sorted(domain_counts.items()):
        print(f"  {d}: {n}")
    print(f"Categories: {sorted(cat_counts.keys())}")
    print(f"Workers: {args.workers}  Max attempts: {args.max_attempts}")
    print(f"Target store: {target_store}\n")

    # -- Process --
    target_store_lock = threading.Lock()

    def process(rec):
        return run_agent_for_record(
            rec=rec,
            data_root=data_root,
            target_store=target_store,
            cfg=cfg,
            target_store_lock=target_store_lock,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, rec): rec for rec in records}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
                mark = "V" if result.get("verified") else "."
                bid = result.get("behavior_id", "?")
                print(f"[{done:4d}/{len(records)}] {mark} {bid}", flush=True)
            except Exception as e:
                rec = futures[fut]
                print(f"[{done:4d}/{len(records)}] ERROR {rec.get('behavior_id','?')}: {e}",
                      flush=True)

    # -- Summary --
    print("\n" + "=" * 60)
    n_canonical = len(list(target_store.glob("*.jpg")))
    print(f"  canonical store: {n_canonical} images in {target_store}")

    for domain in sorted(domain_counts.keys()):
        domain_recs = [r for r in records if r["domain"] == domain]
        cats = sorted(set(r["category"] for r in domain_recs))
        print(f"\n  {domain}:")
        for cat in cats:
            n_cat = sum(1 for r in domain_recs if r["category"] == cat)
            print(f"    {cat:30s}: {n_cat} records")
    print("=" * 60)


if __name__ == "__main__":
    main()
