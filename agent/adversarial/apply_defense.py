"""Apply defenses (PAD / FreqPure / BlueSuffix) to adversarial or clean images.

Supports two modes via --split:
  adversarial  Apply defense to all adversarial attack images (default)
  clean        Apply defense to the 832 clean source images

Outputs:
  adversarial: dataset/realm/defend/<defense>/<attack>/<cat>/<index>.png
  clean:       dataset/realm/defend/clean/<defense>/<cat>/<index>.png

Each attack (or clean split) writes its own manifest.json on completion.
Skip-existing logic allows safe resumption after interruption.

Usage:
    # Adversarial (all attacks, GPU 0)
    python apply_defense.py --defense pad --device cuda:0

    # Adversarial (subset of attacks)
    python apply_defense.py --defense freqpure --attacks advedm vattack --device cuda:1

    # Clean images
    python apply_defense.py --defense bluesuffix --split clean --device cuda:0

    # BlueSuffix without text purifier
    python apply_defense.py --defense bluesuffix --no_text_purifier --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ALL_ATTACKS = [
    "advdiffvlm", "advedm", "advedm_r", "anyattack", "coa",
    "figstep", "foa", "imagemix", "mattack", "paattack",
    "physpatch", "promptinject", "vattack",
]

CATEGORIES = [
    "agibot", "av_meta_actions", "bridgev2", "common_sense",
    "holoassist", "robofail", "robovqa",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_images(cat_dir: Path) -> list[Path]:
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(cat_dir.glob(ext))
    return sorted(imgs)


def load_source_manifest(adv_dir: Path) -> dict:
    p = adv_dir / "manifest.json"
    if not p.exists():
        return {}
    data = json.load(open(p))
    return {s["output_path"]: s for s in data.get("samples", [])}


def load_defense(name: str, args):
    """Initialise and return the requested defense."""
    device = args.device

    if name == "pad":
        from vlm_benchmark.defense.pad.pad_defense import PADDefense, PADDefenseConfig
        cfg = PADDefenseConfig(device=device)
        if getattr(args, "sam_checkpoint", None):
            cfg.sam_checkpoint = args.sam_checkpoint
        d = PADDefense(cfg)
        d._initialize_models()
        return d

    if name == "freqpure":
        from vlm_benchmark.defense.freqpure.freqpure_defense import FreqPureDefense, FreqPureDefenseConfig
        cfg = FreqPureDefenseConfig(device=device)
        if getattr(args, "diffusion_checkpoint", None):
            cfg.diffusion_checkpoint = args.diffusion_checkpoint
        d = FreqPureDefense(cfg)
        d._initialize_models()
        return d

    if name == "bluesuffix":
        from vlm_benchmark.defense.bluesuffix.bluesuffix_defense import BlueSuffixDefense, BlueSuffixDefenseConfig
        cfg = BlueSuffixDefenseConfig(
            device=device,
            enable_image_purifier=not getattr(args, "no_image_purifier", False),
            enable_text_purifier=not getattr(args, "no_text_purifier", False),
            enable_suffix_generator=not getattr(args, "no_suffix_generator", False),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        if getattr(args, "diffusion_checkpoint", None):
            cfg.diffusion_checkpoint = args.diffusion_checkpoint
        if getattr(args, "suffix_generator_dir", None):
            cfg.suffix_generator_dir = args.suffix_generator_dir
        d = BlueSuffixDefense(cfg)
        if cfg.enable_image_purifier:
            d._initialize_image_purifier()
        if cfg.enable_suffix_generator:
            d._initialize_suffix_generator()
        return d

    raise ValueError(f"Unknown defense: {name}. Choose from: pad, freqpure, bluesuffix")


# ── Adversarial mode ──────────────────────────────────────────────────────────

def _adv_entry(defense, attack, cat, img_path, out_file, src_manifest, status, meta=None):
    key = f"{cat}/{img_path.name}"
    src = src_manifest.get(key, {})
    entry = {
        "defense": defense, "attack": attack, "category": cat,
        "index": img_path.stem,
        "adversarial_path": str(img_path), "defended_path": str(out_file),
        "behavior_id": src.get("behavior_id", ""),
        "question": src.get("question", ""),
        "correct_answer": src.get("correct_answer", ""),
        "attack_target": src.get("attack_target", ""),
        "options": src.get("options", {}),
        "status": status,
    }
    if defense == "bluesuffix" and meta:
        entry.update({
            "final_prompt": meta.get("final_prompt", src.get("question", "")),
            "purified_prompt": meta.get("purified_prompt", ""),
            "suffix": meta.get("suffix", ""),
            "steps_applied": meta.get("steps_applied", []),
        })
    return entry


def apply_to_attack(defense_name: str, attack: str, adv_root: Path,
                    out_root: Path, defense) -> dict:
    adv_dir = adv_root / attack
    out_dir = (out_root / attack)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_manifest = load_source_manifest(adv_dir)
    samples, n_done, n_skipped, n_errors = [], 0, 0, 0
    t0 = time.time()

    for cat in CATEGORIES:
        cat_dir = adv_dir / cat
        if not cat_dir.exists():
            continue
        out_cat = out_dir / cat
        out_cat.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(find_images(cat_dir),
                             desc=f"{defense_name}/{attack}/{cat}", leave=False):
            out_file = out_cat / (img_path.stem + ".png")

            if out_file.exists() and out_file.stat().st_size > 0:
                n_skipped += 1
                samples.append(_adv_entry(defense_name, attack, cat,
                                          img_path, out_file, src_manifest, "skipped"))
                continue

            try:
                kwargs = {}
                if defense_name == "bluesuffix":
                    key = f"{cat}/{img_path.name}"
                    kwargs["prompt"] = src_manifest.get(key, {}).get("question", "")
                result = defense.clean(str(img_path), **kwargs)
                result.cleaned_sample.save(str(out_file))
                n_done += 1
                meta = result.metadata if defense_name == "bluesuffix" else {}
                samples.append(_adv_entry(defense_name, attack, cat,
                                          img_path, out_file, src_manifest, "done", meta))
            except Exception as e:
                print(f"\n  ✗ {img_path.name}: {e}")
                n_errors += 1
                samples.append(_adv_entry(defense_name, attack, cat,
                                          img_path, out_file, src_manifest, "error"))

    elapsed = time.time() - t0
    manifest = {
        "defense": defense_name, "attack": attack, "dataset": "realm",
        "n_samples": len(samples), "n_done": n_done,
        "n_skipped": n_skipped, "n_errors": n_errors,
        "elapsed_s": round(elapsed, 1), "samples": samples,
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    return manifest


def run_adversarial(args, defense):
    adv_root = REPO_ROOT / args.adv_root
    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    attacks = [a for a in args.attacks if (adv_root / a).exists()]
    missing = [a for a in args.attacks if not (adv_root / a).exists()]
    if missing:
        print(f"Skipping (not found): {missing}")

    total = sum(
        sum(1 for cat in CATEGORIES
            for _ in find_images(adv_root / a / cat)
            if (adv_root / a / cat).exists())
        for a in attacks
    )
    print(f"\n{args.defense.upper()}: {len(attacks)} attacks, {total} images → {out_root}\n")

    for attack in attacks:
        print(f"\n{'='*60}\nAttack: {attack}")
        m = apply_to_attack(args.defense, attack, adv_root, out_root, defense)
        print(f"  done={m['n_done']} skipped={m['n_skipped']} "
              f"errors={m['n_errors']} in {m['elapsed_s']:.0f}s")

    print("\nAll done.")


# ── Clean mode ────────────────────────────────────────────────────────────────

def _clean_entry(rec, out_file, defense, status, meta=None):
    entry = {
        "defense": defense, "split": "clean",
        "category": rec.output_group, "index": rec.index,
        "behavior_id": rec.behavior_id,
        "source_path": str(rec.source_path),
        "defended_path": str(out_file),
        "question": getattr(rec, "question", ""),
        "correct_answer": getattr(rec, "correct_answer", ""),
        "options": getattr(rec, "options", {}),
        "status": status,
    }
    if defense == "bluesuffix" and meta:
        entry.update({
            "final_prompt": meta.get("final_prompt", ""),
            "purified_prompt": meta.get("purified_prompt", ""),
            "suffix": meta.get("suffix", ""),
        })
    return entry


def run_clean(args, defense):
    clean_root = REPO_ROOT / args.clean_root
    out_root = (REPO_ROOT / args.out_root) / args.defense
    out_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT / "agent"))
    from adversarial.generate import load_realm_attack_records
    records = load_realm_attack_records(clean_root)
    print(f"\n{args.defense.upper()} on clean: {len(records)} records → {out_root}\n")

    source_cache: dict[str, Image.Image] = {}
    samples, n_done, n_skipped, n_errors = [], 0, 0, 0
    t0 = time.time()

    for rec in tqdm(records, desc=f"clean/{args.defense}"):
        out_cat = out_root / rec.output_group
        out_cat.mkdir(parents=True, exist_ok=True)
        out_file = out_cat / f"{rec.index}.png"

        if out_file.exists() and out_file.stat().st_size > 0:
            n_skipped += 1
            samples.append(_clean_entry(rec, out_file, args.defense, "skipped"))
            continue

        src = str(rec.source_path.resolve())
        try:
            if src not in source_cache:
                question = getattr(rec, "question", "")
                kwargs = {"prompt": question} if args.defense == "bluesuffix" else {}
                result = defense.clean(src, **kwargs)
                source_cache[src] = (result.cleaned_sample, getattr(result, "metadata", {}))
            img, meta = source_cache[src]
            img.save(str(out_file))
            n_done += 1
            samples.append(_clean_entry(rec, out_file, args.defense, "done", meta))
        except Exception as e:
            print(f"\n  ✗ {rec.index}: {e}")
            n_errors += 1
            samples.append(_clean_entry(rec, out_file, args.defense, "error"))

    elapsed = time.time() - t0
    manifest = {
        "defense": args.defense, "split": "clean", "dataset": "realm",
        "clean_root": str(clean_root),
        "n_samples": len(samples), "n_done": n_done,
        "n_skipped": n_skipped, "n_errors": n_errors,
        "elapsed_s": round(elapsed, 1), "samples": samples,
    }
    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)
    print(f"\ndone={n_done} skipped={n_skipped} errors={n_errors} in {elapsed:.0f}s")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply a defense to adversarial or clean REALM images."
    )
    parser.add_argument("--defense", required=True, choices=["pad", "freqpure", "bluesuffix"])
    parser.add_argument("--split", choices=["adversarial", "clean"], default="adversarial")
    parser.add_argument("--device", default="cuda:0")

    # Adversarial-mode args
    parser.add_argument("--attacks", nargs="+", default=ALL_ATTACKS)
    parser.add_argument("--adv_root", default="dataset/realm/adversarial")
    parser.add_argument("--out_root", default="dataset/realm/defend")

    # Clean-mode args
    parser.add_argument("--clean_root", default="dataset/realm/clean")

    # PAD
    parser.add_argument("--sam_checkpoint", default=None)

    # FreqPure / BlueSuffix
    parser.add_argument("--diffusion_checkpoint", default=None)

    # BlueSuffix
    parser.add_argument("--suffix_generator_dir", default=None)
    parser.add_argument("--no_image_purifier",   action="store_true")
    parser.add_argument("--no_text_purifier",    action="store_true")
    parser.add_argument("--no_suffix_generator", action="store_true")

    args = parser.parse_args()

    # Adjust out_root for adversarial split to include defense name
    if args.split == "adversarial":
        args.out_root = str(Path(args.out_root) / args.defense)

    defense = load_defense(args.defense, args)

    if args.split == "adversarial":
        run_adversarial(args, defense)
    else:
        run_clean(args, defense)


if __name__ == "__main__":
    main()
