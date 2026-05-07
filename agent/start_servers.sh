#!/usr/bin/env bash
# Launch local model servers on a single B200 GPU (183 GiB VRAM).
# Thinking model uses OpenRouter API (no local VRAM needed).
#
# VRAM budget (183 GiB total):
#   Qwen2.5-VL-7B-Instruct    : 0.15 × 183 ≈ 27 GiB  (verifier)
#   Qwen-Image-2512            : ~64 GiB               (text-to-image generation)
#   Qwen-Image-Edit-2511       : ~64 GiB               (instruction-based editing)
#   Headroom                   : ~28 GiB
#
# Servers:
#   Port 8001 — Qwen2.5-VL-7B-Instruct     (vllm, verification)
#   Port 8091 — Qwen-Image-2512             (vllm-omni, generation)
#   Port 8092 — Qwen-Image-Edit-2511        (vllm-omni, editing)
#
# Thinking model (Steps 1 & 4):
#   OpenRouter API — qwen/qwen3-vl-32b-instruct (or any vision model)
#   Set OPENROUTER_API_KEY before running build_targets.py
#
# Usage:
#   bash agent/start_servers.sh          # start all
#   bash agent/start_servers.sh --stop   # kill all

set -euo pipefail

LOGDIR="./logs"
mkdir -p "$LOGDIR"

VLLM_ENV="./envs/vllm"
OMNI_ENV="./envs/vllm-omni"
CONDA_BASE="/apps/conda/25.7.0"
LMOD_INIT="/apps/lmod/9.1.2/init/bash"

# ── Stop mode ────────────────────────────────────────────────────────────────
kill_server() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] || return 0
    local pid name
    pid=$(<"$pidfile")
    name=$(basename "$pidfile" .pid)
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Killing $name (PID $pid) and children..."
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        pkill -9 -P "$pid" 2>/dev/null || true
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "  $name (PID $pid) already stopped"
    fi
    rm -f "$pidfile"
}

if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping all servers..."
    for pidfile in "$LOGDIR"/*.pid; do
        kill_server "$pidfile"
    done
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm-omni serve" 2>/dev/null || true
    sleep 2
    echo "Done. VRAM freed:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    exit 0
fi

# ── Helper: launch a server in a subshell with its own env ───────────────────
launch_server() {
    local name="$1"
    local env_path="$2"
    local cuda_ver="$3"
    local logfile="$LOGDIR/${name}.log"
    local pidfile="$LOGDIR/${name}.pid"
    shift 3

    kill_server "$pidfile"

    echo "  Starting $name → $logfile"

    bash -c "
        source '$CONDA_BASE/etc/profile.d/conda.sh'
        conda activate '$env_path'
        source '$LMOD_INIT'
        module load cuda/$cuda_ver
        export CUDA_HOME=\$CUDA_ROOT
        export VLLM_VIT_ATTENTION_BACKEND=TORCH_SDPA
        export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
        exec $*
    " > "$logfile" 2>&1 &

    local pid=$!
    echo "$pid" > "$pidfile"
    echo "  $name started (PID $pid)"
}

# ── Helper: wait until a port responds or timeout ────────────────────────────
wait_for_port() {
    local name="$1" port="$2" timeout="${3:-300}"
    local elapsed=0
    echo "  Waiting for $name (port $port)..."
    while ! curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $timeout ]]; then
            echo "  ERROR: $name did not start within ${timeout}s."
            echo "  Check: tail -50 $LOGDIR/${name}.log"
            return 1
        fi
        local pidfile="$LOGDIR/${name}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid=$(<"$pidfile")
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "  ERROR: $name (PID $pid) died during startup."
                echo "  Check: tail -50 $LOGDIR/${name}.log"
                return 1
            fi
        fi
    done
    echo "  $name is ready! (took ~${elapsed}s)"
}

# ── Launch servers ───────────────────────────────────────────────────────────
echo "Launching servers (logs in $LOGDIR/)..."
echo "  Thinking model: OpenRouter API (set OPENROUTER_API_KEY)"
echo ""

# 1) Qwen2.5-VL-7B — verifier
launch_server "qwen25-vl-7b" "$VLLM_ENV" "12.8.1" \
    vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" \
        --port 8001 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 32768 \
        --enforce-eager

wait_for_port "qwen25-vl-7b" 8001 300

# 2) Qwen-Image-2512 — text-to-image generation
launch_server "qwen-image-gen" "$OMNI_ENV" "12.9.1" \
    vllm-omni serve "Qwen/Qwen-Image-2512" \
        --omni \
        --port 8091 \
        --enforce-eager

wait_for_port "qwen-image-gen" 8091 300

# 3) Qwen-Image-Edit-2511 — instruction-based image editing
launch_server "qwen-image-edit" "$OMNI_ENV" "12.9.1" \
    vllm-omni serve "Qwen/Qwen-Image-Edit-2511" \
        --omni \
        --port 8092 \
        --enforce-eager

wait_for_port "qwen-image-edit" 8092 300

echo ""
echo "All servers ready!"
echo ""
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""
echo "Endpoints:"
echo "  Thinking:   OpenRouter API (qwen/qwen3-vl-32b-instruct)"
echo "  Verifier:   http://localhost:8001/v1  — Qwen2.5-VL-7B-Instruct"
echo "  Generation: http://localhost:8091/v1  — Qwen-Image-2512"
echo "  Editing:    http://localhost:8092/v1  — Qwen-Image-Edit-2511"
echo ""
echo "Run target generation:"
echo "  python -m agent.target_generation.build_targets \\"
echo "      --data_root dataset/physbench-verified \\"
echo "      --domains physbench \\"
echo "      --gen_server http://localhost:8091 \\"
echo "      --edit_server http://localhost:8092 \\"
echo "      --vllm_server http://localhost:8001 \\"
echo "      --max_attempts 3 --workers 3"
echo ""
echo "Stop all:  bash agent/start_servers.sh --stop"
