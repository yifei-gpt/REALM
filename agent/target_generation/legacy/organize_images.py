#!/usr/bin/env python3
"""
Organize clean images into per-category directories with symlinks and manifests.

Reads behaviors.jsonl, creates:
  {output_dir}/{category}/source/{NNN}.jpg  (symlink to original image)
  {output_dir}/{category}/manifest.json     (record metadata)

Usage:
    python -m agent.target_generation.organize_images \
        --behaviors dataset/redteam/behaviors.jsonl \
        --output_dir dataset/redteam
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, DATASET_ROOT


def main():
    parser = argparse.ArgumentParser(description="Organize clean images by category")
    parser.add_argument(
        "--behaviors",
        default=str(DATASET_ROOT / "behaviors.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATASET_ROOT),
    )
    args = parser.parse_args()

    behaviors_path = Path(args.behaviors)
    output_dir = Path(args.output_dir)

    records = [json.loads(line) for line in behaviors_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} records from {behaviors_path}")

    by_cat = defaultdict(list)
    for rec in records:
        by_cat[rec["category"]].append(rec)

    for cat in sorted(by_cat):
        cat_records = by_cat[cat]
        src_dir = output_dir / cat / "source"
        src_dir.mkdir(parents=True, exist_ok=True)

        manifest_records = []
        for i, rec in enumerate(cat_records):
            idx = f"{i:03d}"
            dst = src_dir / f"{idx}.jpg"

            # Resolve original image path
            img_rel = rec["images"][0]
            img_abs = Path(img_rel) if Path(img_rel).is_absolute() else PROJECT_ROOT / img_rel
            if not img_abs.exists():
                print(f"  WARNING: image not found: {img_abs}")
                continue

            # Create symlink
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(img_abs.resolve())

            manifest_records.append({
                "index": idx,
                "behavior_id": rec["behavior_id"],
                "question": rec["question"],
                "correct_answer": rec["correct_answer"],
                "attack_target_text": rec["attack_target_text"],
                "qa_format": rec["qa_format"],
                "compatible_attacks": rec["compatible_attacks"],
                "source_path": str(dst),
                "original_image_path": str(img_abs),
            })

        manifest = {
            "category": cat,
            "n": len(manifest_records),
            "records": manifest_records,
        }
        manifest_path = output_dir / cat / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Verify symlinks resolve
        broken = sum(1 for r in manifest_records if not Path(r["source_path"]).resolve().exists())
        status = f" ({broken} broken symlinks!)" if broken else ""
        print(f"  {cat:<25} {len(manifest_records):>4} images{status}")

    # Copy behaviors files into output dir (if not already there)
    for name in ["behaviors.jsonl", "behaviors_raw.jsonl"]:
        src = behaviors_path.parent / name
        dst = output_dir / name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    print(f"\nDone. {len(by_cat)} categories in {output_dir}")


if __name__ == "__main__":
    main()
