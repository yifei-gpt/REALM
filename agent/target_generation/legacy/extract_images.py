"""
One-time image extraction for the red-teaming candidate dataset.

Reads behaviors_raw.jsonl, extracts/links all images into
dataset/redteam/images/, and updates image paths in-place.

Usage:
    python -m agent.target_generation.extract_images \
        --behaviors dataset/redteam/behaviors_raw.jsonl \
        --output_dir dataset/redteam/images \
        --data_root dataset/general
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, DATASET_ROOT


# ---------------------------------------------------------------------------
# NuScenes-QA — build sample_token -> CAM_FRONT filename mapping
# ---------------------------------------------------------------------------

def build_nuscenes_cam_map(data_root: str) -> dict:
    try:
        import ijson
    except ImportError:
        raise RuntimeError("ijson required: pip install ijson")

    sd_path = os.path.join(
        data_root, "NuScenes-QA/data/v1.0-trainval/sample_data.json"
    )
    cam_map = {}
    print(f"  Streaming sample_data.json ({os.path.getsize(sd_path) // 1_000_000} MB)…")
    with open(sd_path, "rb") as f:
        for rec in ijson.items(f, "item"):
            if rec.get("is_key_frame") and "__CAM_FRONT__" in rec.get("filename", ""):
                cam_map[rec["sample_token"]] = rec["filename"]
    print(f"  cam_map: {len(cam_map)} entries")
    return cam_map


# ---------------------------------------------------------------------------
# Symlink helper (falls back to copy if needed)
# ---------------------------------------------------------------------------

def ensure_link(src: str, dst: str) -> None:
    """Create symlink dst -> src, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        return
    if os.path.exists(src):
        os.symlink(os.path.abspath(src), dst)
    else:
        print(f"  WARNING: source not found: {src}")


# ---------------------------------------------------------------------------
# drive-action image extraction
# ---------------------------------------------------------------------------

def extract_drive_action_images(behaviors: list, data_root: str, output_dir: str) -> dict:
    """
    For drive_action records, extract JPEG bytes from parquet and save to disk.
    Returns {question_slice_id: [path0, path1, path2]} map.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required: pip install pandas pyarrow")

    dest_dir = os.path.join(output_dir, "drive_action")
    os.makedirs(dest_dir, exist_ok=True)

    # Collect which (parquet_file, row_index, question_slice_id) we need
    needed = {}  # parquet_file -> [(row_index, question_slice_id)]
    for b in behaviors:
        if b["source_dataset"] != "drive_action":
            continue
        meta = b["source_metadata"]
        pf = meta["parquet_file"]
        ri = meta["row_index"]
        qid = meta["question_slice_id"]
        needed.setdefault(pf, []).append((ri, qid))

    if not needed:
        return {}

    parquet_dir = os.path.join(data_root, "drive-action/data")
    extracted = {}  # qid -> [path0, path1, path2]

    for parquet_file, items in needed.items():
        path = os.path.join(parquet_dir, parquet_file)
        print(f"  Reading {parquet_file} ({len(items)} rows to extract)…")
        df = pd.read_parquet(path)

        row_map = {ri: qid for ri, qid in items}
        for ri, qid in row_map.items():
            paths = []
            for col_idx, col in enumerate(["image_0", "image_1", "image_2"]):
                out_path = os.path.join(dest_dir, f"{qid}_{col_idx}.jpg")
                if not os.path.exists(out_path):
                    img_dict = df.iloc[ri][col]
                    if isinstance(img_dict, dict):
                        img_bytes = img_dict.get("bytes")
                    else:
                        img_bytes = None
                    if img_bytes:
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                    else:
                        print(f"  WARNING: no bytes for {qid} col {col}")
                paths.append(out_path)
            extracted[qid] = paths

    print(f"  Extracted {len(extracted)} drive-action image sets")
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract/link images for red-teaming dataset")
    parser.add_argument(
        "--behaviors",
        default=str(DATASET_ROOT / "behaviors_raw.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATASET_ROOT / "images"),
    )
    parser.add_argument(
        "--data_root",
        default=str(PROJECT_ROOT / "dataset" / "general"),
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking"
    )
    args = parser.parse_args()

    # Load behaviors
    behaviors = []
    with open(args.behaviors) as f:
        for line in f:
            behaviors.append(json.loads(line))
    print(f"Loaded {len(behaviors)} behavior records from {args.behaviors}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- NuScenes-QA: symlink CAM_FRONT images ---
    ns_dir = os.path.join(args.output_dir, "nuscenes_qa")
    os.makedirs(ns_dir, exist_ok=True)
    ns_base = os.path.join(args.data_root, "NuScenes-QA/data")

    nuscenes_linked = 0
    for b in behaviors:
        if b["source_dataset"] != "nuscenes_qa":
            continue
        for img_path in b["images"]:
            # img_path is like: dataset/general/NuScenes-QA/data/samples/CAM_FRONT/xxx.jpg
            # Extract just the filename
            fname = os.path.basename(img_path)
            dst = os.path.join(ns_dir, fname)
            src = os.path.join(ns_base, "samples/CAM_FRONT", fname)
            ensure_link(src, dst)
            nuscenes_linked += 1
    print(f"NuScenes-QA: {nuscenes_linked} image links created in {ns_dir}")

    # --- DriveBench: symlink all referenced images ---
    db_dir = os.path.join(args.output_dir, "drivebench")
    db_base = os.path.join(args.data_root, "DriveBench")

    db_linked = 0
    for b in behaviors:
        if b["source_dataset"] != "drivebench":
            continue
        for img_path in b["images"]:
            # img_path: dataset/general/DriveBench/data/nuscenes/samples/CAM_XXX/xxx.jpg
            fname = os.path.basename(img_path)
            # Preserve camera subfolder
            cam_folder = os.path.basename(os.path.dirname(img_path))
            cam_dir = os.path.join(db_dir, cam_folder)
            os.makedirs(cam_dir, exist_ok=True)
            dst = os.path.join(cam_dir, fname)
            # src relative to data_root: strip "dataset/general/DriveBench/"
            rel = img_path.replace("dataset/general/DriveBench/", "")
            src = os.path.join(db_base, rel)
            ensure_link(src, dst)
            db_linked += 1
    print(f"DriveBench: {db_linked} image links created in {db_dir}")

    # --- drive-action: extract from parquet ---
    print("drive-action: extracting images from parquet…")
    extracted = extract_drive_action_images(behaviors, args.data_root, args.output_dir)
    print(f"drive-action: done ({len(extracted)} QA image sets)")

    # --- Update image paths in behaviors_raw.jsonl ---
    print("Updating image paths in behaviors_raw.jsonl…")
    updated_behaviors = []
    for b in behaviors:
        if b["source_dataset"] == "nuscenes_qa":
            new_images = []
            for img_path in b["images"]:
                fname = os.path.basename(img_path)
                new_images.append(os.path.join(args.output_dir, "nuscenes_qa", fname))
            b["images"] = new_images
        elif b["source_dataset"] == "drivebench":
            new_images = []
            for img_path in b["images"]:
                fname = os.path.basename(img_path)
                cam_folder = os.path.basename(os.path.dirname(img_path))
                new_images.append(os.path.join(args.output_dir, "drivebench", cam_folder, fname))
            b["images"] = new_images
        elif b["source_dataset"] == "drive_action":
            qid = b["source_metadata"]["question_slice_id"]
            if qid in extracted:
                b["images"] = extracted[qid]
        updated_behaviors.append(b)

    # Write updated JSONL
    with open(args.behaviors, "w") as f:
        for b in updated_behaviors:
            f.write(json.dumps(b) + "\n")
    print(f"Updated {args.behaviors} with resolved image paths")

    # Verify
    missing = 0
    for b in updated_behaviors:
        for img in b["images"]:
            if not os.path.exists(img):
                missing += 1
    print(f"\nVerification: {missing} missing image files out of {sum(len(b['images']) for b in updated_behaviors)} total")


if __name__ == "__main__":
    main()
