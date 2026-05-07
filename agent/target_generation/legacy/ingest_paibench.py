"""
Ingest PAI-bench records into behaviors.jsonl.

Replaces the manual process of adding PAI-bench records. Reads the raw
parquet, matches records to extracted video-frame images, assigns
deterministic wrong answers, and optionally pre-filters with a VLM.

Usage:
    python -m agent.target_generation.ingest_paibench \
        --parquet_path normal_inference/physical-ai-bench/understanding/data/data/test-00000-of-00001.parquet \
        --image_dir dataset/redteam/pai_bench \
        --output dataset/redteam/behaviors.jsonl

    # With VLM prefilter:
    python -m agent.target_generation.ingest_paibench \
        --parquet_path ... --image_dir ... --output ... \
        --vllm_server http://localhost:8001/v1 \
        --vlm_model Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATASET_ROOT,
    PAIBENCH_SUBCAT_DIR,
    format_paibench_question,
    get_attack_target_paibench,
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    """Encode an image file as a base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{data}"


def resolve_image(video_path: str, image_dir: Path) -> Path | None:
    """Map a parquet video_path to the corresponding extracted frame image.

    Image filename = video filename stem + .jpg
    Directory = image_dir / subcat_dir (derived from the video_path directory).
    """
    vp = Path(video_path)
    image_name = vp.stem + ".jpg"

    # The video_path dir may have a date suffix (e.g. av_meta_actions_20250227).
    # The image dirs use the canonical names from PAIBENCH_SUBCAT_DIR values.
    video_dir = vp.parent.name  # e.g. "av_meta_actions_20250227"

    # Find which canonical subcat_dir this video_dir maps to
    canonical_dirs = set(PAIBENCH_SUBCAT_DIR.values())
    for cdir in canonical_dirs:
        if video_dir.startswith(cdir):
            img_path = image_dir / cdir / image_name
            if img_path.exists():
                return img_path
            break

    # Fallback: search all subdirs
    for cdir in canonical_dirs:
        img_path = image_dir / cdir / image_name
        if img_path.exists():
            return img_path

    return None


def canonical_video_path(video_path: str) -> str:
    """Normalize video_path to match existing behaviors.jsonl format.

    Strips date suffixes from directory names, e.g.:
        videos/av_meta_actions_20250227/uuid.mp4 -> videos/av_meta_actions/uuid.mp4
    """
    vp = Path(video_path)
    video_dir = vp.parent.name
    canonical_dirs = set(PAIBENCH_SUBCAT_DIR.values())
    for cdir in canonical_dirs:
        if video_dir.startswith(cdir) and video_dir != cdir:
            return str(vp.parent.parent / cdir / vp.name)
    return video_path


# ---------------------------------------------------------------------------
# VLM prefilter
# ---------------------------------------------------------------------------

class VLMPrefilter:
    """Query a vLLM server to check if VLM answers correctly on clean images."""

    def __init__(self, server_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=server_url, api_key="dummy")
        self.model = model

    def answers_correctly(self, image_path: str, question: str,
                          options: dict, correct_letter: str) -> bool:
        """Return True if VLM picks the correct answer on the clean image."""
        formatted_q = format_paibench_question(question, options)
        data_url = encode_image(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": formatted_q},
        ]}]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=32,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content or ""
            # Check if VLM picked the correct letter
            return answer.strip().upper().startswith(correct_letter.upper())
        except Exception as e:
            print(f"  VLM error: {e}")
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PAI-bench records into behaviors.jsonl"
    )
    parser.add_argument(
        "--parquet_path",
        default=str(
            Path("normal_inference/physical-ai-bench/understanding"
                 "/data/data/test-00000-of-00001.parquet")
        ),
        help="Path to PAI-bench parquet file",
    )
    parser.add_argument(
        "--image_dir",
        default=str(DATASET_ROOT / "pai_bench"),
        help="Directory containing extracted PAI-bench video frame images",
    )
    parser.add_argument(
        "--output",
        default=str(DATASET_ROOT / "behaviors.jsonl"),
        help="Output behaviors.jsonl path (append mode)",
    )
    parser.add_argument("--vllm_server", default=None, help="vLLM server URL for prefilter")
    parser.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent VLM workers")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    # ---- Step 1: Read parquet ----
    print(f"Reading parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    print(f"  Total records in parquet: {len(df)}")

    if args.limit:
        df = df.head(args.limit)
        print(f"  Limited to first {args.limit} records")

    # ---- Step 2: Load existing behavior IDs to skip duplicates ----
    existing_ids: set[str] = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # PAI-bench records may not have behavior_id; deduplicate by
                # (domain, image) tuple instead
                if rec.get("domain") == "pai-bench":
                    existing_ids.add(rec.get("image", ""))
        print(f"  Existing PAI-bench images in output: {len(existing_ids)}")

    # ---- Step 3: Match records to images and build candidates ----
    candidates = []
    skipped_no_image = 0
    skipped_no_subcat = 0
    skipped_no_target = 0
    skipped_duplicate = 0

    for _, row in df.iterrows():
        subcategory = row["subcategory"]
        video_path = row["video_path"]
        options = row["index2ans"]
        correct_letter = row["answer"]
        question = row["question"]
        category = row["category"]

        # Check subcategory is known
        if subcategory not in PAIBENCH_SUBCAT_DIR:
            skipped_no_subcat += 1
            continue

        # Resolve image
        img_path = resolve_image(video_path, image_dir)
        if img_path is None:
            skipped_no_image += 1
            continue

        image_name = img_path.name  # e.g. uuid.jpg

        # Skip duplicates
        if image_name in existing_ids:
            skipped_duplicate += 1
            continue

        # Determine attack target
        attack_target = get_attack_target_paibench(options, correct_letter)
        if not attack_target:
            skipped_no_target += 1
            continue

        # Normalize video_path (strip date suffixes)
        norm_video_path = canonical_video_path(video_path)

        candidates.append({
            "image": image_name,
            "video_path": norm_video_path,
            "question": question,
            "options": options,
            "answer": correct_letter,
            "category": category,
            "subcategory": subcategory,
            "domain": "pai-bench",
            # Keep resolved path for VLM prefilter (not written to output)
            "_image_path": str(img_path),
        })

    print(f"\n  Candidates with matching images: {len(candidates)}")
    print(f"  Skipped (no image):     {skipped_no_image}")
    print(f"  Skipped (unknown subcat): {skipped_no_subcat}")
    print(f"  Skipped (no target):    {skipped_no_target}")
    print(f"  Skipped (duplicate):    {skipped_duplicate}")

    if not candidates:
        print("No candidates to process. Exiting.")
        return

    # ---- Step 4: Optional VLM prefilter ----
    if args.vllm_server:
        print(f"\nRunning VLM prefilter ({args.vllm_server}, model={args.vlm_model})…")
        prefilter = VLMPrefilter(args.vllm_server, args.vlm_model)
        passed = []
        failed = 0
        lock = threading.Lock()
        total = len(candidates)
        done = 0
        t0 = time.time()

        def check_one(cand: dict) -> dict | None:
            nonlocal done, failed
            ok = prefilter.answers_correctly(
                cand["_image_path"],
                cand["question"],
                cand["options"],
                cand["answer"],
            )
            with lock:
                done += 1
                if not ok:
                    failed += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{total}] passed={done - failed} "
                          f"failed={failed} rate={rate:.1f}/s")
            return cand if ok else None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(check_one, c) for c in candidates]
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    passed.append(result)

        print(f"  VLM prefilter: {len(passed)}/{total} passed")
        candidates = passed

    # ---- Step 5: Write output ----
    # Remove internal fields before writing
    for cand in candidates:
        cand.pop("_image_path", None)

    # Append mode: add to existing file
    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode) as f:
        for cand in candidates:
            f.write(json.dumps(cand) + "\n")

    # Print summary
    subcat_counts = Counter(c["subcategory"] for c in candidates)
    print(f"\nWrote {len(candidates)} PAI-bench records to {args.output}")
    print(f"Subcategory breakdown:")
    for subcat in sorted(subcat_counts):
        print(f"  {subcat:<25} {subcat_counts[subcat]:>4}")


if __name__ == "__main__":
    main()
