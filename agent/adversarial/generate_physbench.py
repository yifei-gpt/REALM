#!/usr/bin/env python3
"""
Generate adversarial images for PhysBench single-image records (1,799 MCQ questions).

Supports both original PhysBench (test.json) and physbench-verified (manifest.json).

Compatible attacks (no target images needed):
  - Untargeted:    paattack, corruption
  - Text-guided:   vattack, advedm_r (uses wrong answer text as target)
  - Typographic:   figstep (text overlay on source image)
  - Prompt manip:  promptinject (modifies question text)
  - Targeted:      foa, mattack, etc. — NOT supported (no target images)

Usage:
    # First run build_targets.py to generate agent-chosen attack targets, then:
    python generate_physbench.py paattack -o dataset/physbench_adversarial/paattack
    python generate_physbench.py corruption -o dataset/physbench_adversarial/corruption
    python generate_physbench.py vattack -o dataset/physbench_adversarial/vattack

    # Attack targets are loaded from {physbench_root}/target_images/manifest.json
    # Records without an agent-chosen target are skipped.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vlm_benchmark.attacks.registry import register_all_attacks, AttackRegistry
from vlm_benchmark.data.base_dataset import Sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHYSBENCH_ROOT_DEFAULT = PROJECT_ROOT / "dataset" / "physbench-verified"

# Attacks compatible with PhysBench (no target images)
PHYSBENCH_ATTACKS = [
    "paattack", "corruption", "vattack", "advedm_r",
    "figstep", "promptinject",
]
CORRUPTION_MODES = ["brightness", "fog", "lowlight", "motionblur", "watersplash", "saturate"]
PNG_ATTACKS = {"advedm_r", "paattack"}
TEXT_GUIDED_ATTACKS = {"figstep", "promptinject"}

DEFAULT_SMART_MODEL = "Qwen/Qwen3-4B"
VLLM_STARTUP_TIMEOUT = 600


# ---------------------------------------------------------------------------
# PhysBench data loading
# ---------------------------------------------------------------------------

def load_target_manifest(target_manifest_path: Path) -> dict:
    """Load agent-chosen attack targets from target_images/manifest.json.

    Returns {behavior_id: attack_target_text}.
    """
    if not target_manifest_path.exists():
        sys.exit(f"ERROR: target manifest not found: {target_manifest_path}\n"
                 f"Run build_targets.py first to generate agent-chosen attack targets.")
    data = json.loads(target_manifest_path.read_text())
    index = {}
    for entry in data.get("records", []):
        bid = entry.get("behavior_id", "")
        target = entry.get("attack_target", "")
        if bid and target:
            index[bid] = target
    return index


def load_physbench_records(physbench_root: Path, target_manifest: Path,
                           task_types=None, max_samples=None):
    """Load single-image PhysBench records with agent-chosen attack targets.

    Supports two formats:
      1. Original: test.json + test_answer.json + image/ directory
      2. Verified: manifest.json + {category}/source/ directories

    Attack targets are loaded from the target manifest (built by build_targets.py).
    """
    # Load agent-chosen targets
    target_index = load_target_manifest(target_manifest)

    manifest_path = physbench_root / "manifest.json"
    test_path = physbench_root / "test.json"

    records = []
    n_missing_target = 0

    # Format 2: physbench-verified (manifest.json with records)
    if manifest_path.exists() and not test_path.exists():
        data = json.loads(manifest_path.read_text())
        for r in data.get("records", []):
            img_path = physbench_root / r["frame_path"]
            if not img_path.exists():
                continue
            correct_letter = r.get("answer", "")
            if correct_letter not in ("A", "B", "C", "D"):
                continue

            options = r.get("options", {})
            task_type = r.get("category", "")
            if task_types and task_type not in task_types:
                continue

            # Look up agent-chosen attack target
            bid = r.get("behavior_id", f"physbench_{task_type}_{r.get('subcategory', '')}_{r['doc_id']}")
            attack_target = target_index.get(bid, "")
            if not attack_target:
                n_missing_target += 1
                continue

            records.append({
                "idx": r["doc_id"],
                "image_path": img_path,
                "question": r["question"],
                "options": options,
                "correct_letter": correct_letter,
                "correct_answer": options.get(correct_letter, ""),
                "attack_target": attack_target,
                "task_type": task_type,
                "sub_type": r.get("subcategory", ""),
                "ability_type": r.get("ability_type", ""),
            })

    else:
        # Format 1: original PhysBench (test.json + test_answer.json)
        test_data = json.loads(test_path.read_text())
        answers = json.loads((physbench_root / "test_answer.json").read_text())
        ans_index = {r['idx']: r for r in answers}

        for r in test_data:
            files = r['file_name']
            imgs = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
            vids = [f for f in files if f.endswith('.mp4')]
            if len(imgs) != 1 or len(vids) != 0:
                continue

            img_path = physbench_root / "image" / imgs[0]
            if not img_path.exists():
                continue

            ans = ans_index.get(r['idx'], {})
            correct_letter = ans.get('answer', '')
            if correct_letter not in ('A', 'B', 'C', 'D'):
                continue

            # Parse options from question
            options = {}
            for m in re.finditer(r'([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)', r['question'], re.DOTALL):
                options[m.group(1)] = m.group(2).strip()

            task_type = ans.get('task_type', '')
            if task_types and task_type not in task_types:
                continue

            # Look up agent-chosen attack target
            bid = f"physbench_{task_type}_{ans.get('sub_type', '')}_{r['idx']}"
            attack_target = target_index.get(bid, "")
            if not attack_target:
                n_missing_target += 1
                continue

            records.append({
                "idx": r['idx'],
                "image_path": img_path,
                "question": r['question'],
                "options": options,
                "correct_letter": correct_letter,
                "correct_answer": options.get(correct_letter, ''),
                "attack_target": attack_target,
                "task_type": task_type,
                "sub_type": ans.get('sub_type', ''),
                "ability_type": ans.get('ability_type', ''),
            })

    if n_missing_target > 0:
        print(f"WARNING: {n_missing_target} records skipped — no agent-chosen target in manifest")

    if max_samples:
        records = records[:max_samples]

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="physbench_attack_tmp_")
    atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
    return d


def make_sample(rec: dict) -> Sample:
    return Sample(
        id=str(rec['idx']),
        images=[Image.open(rec['image_path']).convert("RGB")],
        question=rec['question'],
        ground_truth=rec['correct_answer'],
        task_type=rec['task_type'],
        metadata={
            "idx": rec['idx'],
            "attack_target_text": rec['attack_target'],
            "attack_source_text": rec['correct_answer'],
            "sub_type": rec['sub_type'],
            "ability_type": rec['ability_type'],
        },
    )


def make_manifest_entry(rec: dict, out_file: Path, out_dir: Path,
                        status: str, perturbation_norm=None,
                        error=None, attack_metadata=None) -> dict:
    entry = {
        "idx": rec['idx'],
        "task_type": rec['task_type'],
        "sub_type": rec['sub_type'],
        "ability_type": rec['ability_type'],
        "source_path": str(rec['image_path']),
        "output_path": str(out_file.relative_to(out_dir)) if out_file.exists() else "",
        "question": rec['question'],
        "correct_answer": rec['correct_answer'],
        "correct_letter": rec['correct_letter'],
        "attack_target": rec['attack_target'],
        "options": rec['options'],
        "status": status,
    }
    if perturbation_norm is not None:
        entry["perturbation_norm"] = perturbation_norm
    if error is not None:
        entry["error"] = error
    if attack_metadata:
        entry["attack_metadata"] = attack_metadata
    return entry


# ---------------------------------------------------------------------------
# Auto vLLM server (for figstep/promptinject smart mode)
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    endpoint = f"{url}/v1/models"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(endpoint, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def start_vllm_server(model: str, device: str):
    port = _find_free_port()
    gpu_id = device.split(":")[1] if ":" in device else "0"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.6"]
    log_file = tempfile.NamedTemporaryFile(prefix="vllm_smart_", suffix=".log",
                                           delete=False, mode="w")
    print(f"Starting vLLM: {model} on GPU {gpu_id}, port {port}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    atexit.register(lambda: (proc.terminate() if proc.poll() is None else None))
    url = f"http://localhost:{port}"
    if not _wait_for_server(url):
        proc.terminate()
        sys.exit(f"ERROR: vLLM failed to start. Log: {log_file.name}")
    print(f"  Ready: {url}")
    return proc, url


def _vlm_chat(client, model: str, prompt: str, max_tokens: int = 256) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant. /no_think"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens, temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip().strip("\"'.")
    except Exception as e:
        print(f"WARNING: VLM call failed ({e}), using fallback")
        return ""


def smart_generate_injection(client, model, attack_name, target_text,
                             question="", level="instruction"):
    if attack_name == "figstep":
        return _vlm_chat(client, model,
            f'Generate a realistic image annotation label describing "{target_text}".\n'
            f"Write a short caption (5-10 words). Reply with ONLY the label.")
    elif attack_name == "promptinject":
        style = "authoritative override" if level == "authority" else "subtle hint"
        return _vlm_chat(client, model,
            f'Generate a {style} directing the response toward "{target_text}".\n'
            f"Under 25 words. Reply with ONLY the text.")
    return None


# ---------------------------------------------------------------------------
# Attack creation
# ---------------------------------------------------------------------------

def create_attack(name: str, device: str, **extra):
    kwargs = {"device": device}
    if name == "paattack":
        kwargs.setdefault("epsilon", 16.0 / 255.0)
        kwargs.setdefault("max_iterations", 300)
    elif name == "vattack":
        kwargs.setdefault("epsilon", 16.0)
        kwargs.setdefault("max_iterations", 300)
    elif name == "advedm_r":
        kwargs.setdefault("target_text", "placeholder")
    elif name == "promptinject":
        kwargs.setdefault("level", "instruction")
    kwargs.update(extra)
    return AttackRegistry.create(name, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate adversarial images for PhysBench single-image records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("attack", choices=PHYSBENCH_ATTACKS, help="Attack name")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--physbench_root", default=str(PHYSBENCH_ROOT_DEFAULT))
    p.add_argument("--task_types", nargs="+", default=None,
                   help="Filter: property, dynamics, relationships")
    p.add_argument("--max_samples", type=int)
    p.add_argument("--device", default="cuda")

    # Overrides
    p.add_argument("--epsilon", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--corruption_mode", default="fog", choices=CORRUPTION_MODES)
    p.add_argument("--corruption_severity", type=int, default=3, choices=[1,2,3,4,5])
    p.add_argument("--level", choices=["instruction", "authority"], default="instruction")

    # Smart mode
    p.add_argument("--vlm_url")
    p.add_argument("--vlm_model", default=DEFAULT_SMART_MODEL)
    p.add_argument("--vlm_device")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    physbench_root = Path(args.physbench_root)

    target_manifest = physbench_root / "target_images" / "manifest.json"
    records = load_physbench_records(
        physbench_root, target_manifest,
        args.task_types, args.max_samples)
    if not records:
        sys.exit(f"ERROR: no records found in {physbench_root}")

    task_types = sorted(set(r['task_type'] for r in records))
    print(f"Attack:       {args.attack}")
    print(f"PhysBench:    {physbench_root}")
    print(f"Records:      {len(records)}")
    print(f"Task types:   {task_types}")
    print(f"Output:       {args.output}\n")

    register_all_attacks()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png" if args.attack in PNG_ATTACKS else ".jpg"

    extra = {}
    if args.epsilon is not None:
        extra["epsilon"] = args.epsilon
    if args.steps is not None:
        extra["max_iterations"] = args.steps
    if args.attack == "corruption":
        extra["severity"] = args.corruption_severity
    if args.attack == "promptinject":
        extra["level"] = args.level

    # Smart mode for figstep/promptinject
    smart_client = None
    if args.attack in TEXT_GUIDED_ATTACKS:
        vlm_url = args.vlm_url
        if not vlm_url:
            vlm_device = args.vlm_device or args.device
            _, vlm_url = start_vllm_server(args.vlm_model, vlm_device)
        from openai import OpenAI
        smart_client = OpenAI(base_url=f"{vlm_url}/v1", api_key="dummy")

    from tqdm import tqdm
    samples_out = []
    n_errors = 0
    t0 = datetime.now()

    # === Corruption: all 6 modes ===
    if args.attack == "corruption":
        severity = extra.get("severity", 3)
        for mode in CORRUPTION_MODES:
            mode_dir = out_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            attack = create_attack("corruption", args.device,
                                   mode=mode, severity=severity)
            mode_samples = []
            mode_errors = 0
            for rec in tqdm(records, desc=f"corruption/{mode}"):
                out_file = mode_dir / rec['task_type'] / f"{rec['idx']}{ext}"
                if out_file.exists() and out_file.stat().st_size > 0:
                    mode_samples.append(make_manifest_entry(
                        rec, out_file, mode_dir, "skipped"))
                    continue
                try:
                    sample = make_sample(rec)
                    result = attack.generate(model=None, sample=sample)
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    result.adversarial_sample.save(out_file)
                    mode_samples.append(make_manifest_entry(
                        rec, out_file, mode_dir, "done", perturbation_norm=0.0))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    mode_errors += 1
                    mode_samples.append(make_manifest_entry(
                        rec, out_file, mode_dir, "failed", error=str(e)))
            manifest = {
                "attack": f"corruption_{mode}", "dataset": "physbench",
                "mode": mode, "severity": severity,
                "n_samples": len(mode_samples),
                "n_done": sum(1 for s in mode_samples if s["status"] == "done"),
                "n_errors": mode_errors,
                "generated_at": datetime.now().isoformat(),
                "samples": mode_samples,
            }
            with open(mode_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"  {mode}: {manifest['n_done']}/{len(records)} ({mode_errors} errors)")
        print(f"\nDone: all 6 modes in {(datetime.now()-t0).total_seconds():.1f}s → {out_dir}")
        return

    # === Regular attacks ===
    attack = create_attack(args.attack, args.device, **extra)

    for rec in tqdm(records, desc=args.attack):
        out_file = out_dir / rec['task_type'] / f"{rec['idx']}{ext}"

        if out_file.exists() and out_file.stat().st_size > 0:
            samples_out.append(make_manifest_entry(rec, out_file, out_dir, "skipped"))
            continue

        try:
            sample = make_sample(rec)

            # Smart mode injection text
            if smart_client and rec['attack_target']:
                injection = smart_generate_injection(
                    smart_client, args.vlm_model, args.attack,
                    rec['attack_target'], question=rec['question'],
                    level=args.level)
                if injection:
                    sample.metadata["attack_injection_text"] = injection

            # advedm_r: update target_text per sample
            if args.attack == "advedm_r":
                attack.config.target_text = rec['correct_answer']
                attack._attack_objs = None

            # vattack: update source/target text per sample
            if args.attack == "vattack":
                attack.config.source_text = rec['correct_answer']
                attack.config.target_text = rec['attack_target']

            result = attack.generate(model=None, sample=sample)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            result.adversarial_sample.save(out_file)

            atk_meta = None
            if result.metadata:
                meta = result.metadata
                if args.attack == "figstep":
                    atk_meta = {
                        "injection_text": meta.get("injection_text", ""),
                        "text_prompt": meta.get("text_prompt", ""),
                    }
                elif args.attack == "promptinject":
                    atk_meta = {
                        "injection_text": meta.get("injection_text", ""),
                        "adversarial_question": meta.get("adversarial_question", ""),
                    }

            samples_out.append(make_manifest_entry(
                rec, out_file, out_dir, "done",
                perturbation_norm=result.perturbation_norm,
                attack_metadata=atk_meta))
        except Exception as e:
            import traceback; traceback.print_exc()
            n_errors += 1
            samples_out.append(make_manifest_entry(
                rec, out_file, out_dir, "failed", error=str(e)))

    duration = (datetime.now() - t0).total_seconds()
    manifest = {
        "attack": args.attack,
        "dataset": "physbench",
        "n_samples": len(samples_out),
        "n_done": sum(1 for s in samples_out if s["status"] == "done"),
        "n_skipped": sum(1 for s in samples_out if s["status"] == "skipped"),
        "n_errors": n_errors,
        "duration_s": round(duration, 1),
        "generated_at": datetime.now().isoformat(),
        "samples": samples_out,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_ok = manifest["n_done"] + manifest["n_skipped"]
    print(f"\nDone: {n_ok}/{len(records)} ({n_errors} errors) in {duration:.1f}s")
    print(f"Output:   {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
