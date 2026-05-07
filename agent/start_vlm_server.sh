#!/usr/bin/env bash
# Start a vLLM VLM server for evaluation on a single B200 GPU (192 GB VRAM).
#
# Usage:
#   bash start_vlm_server.sh qwen3vl8b                # Qwen/Qwen3-VL-8B-Instruct
#   bash start_vlm_server.sh qwen3vl30b               # Qwen/Qwen3-VL-30B-A3B-Instruct
#   bash start_vlm_server.sh qwen3vl32b               # Qwen/Qwen3-VL-32B-Instruct
#   bash start_vlm_server.sh qwen35-9b                # Qwen/Qwen3.5-9B
#   bash start_vlm_server.sh qwen35-27b               # Qwen/Qwen3.5-27B
#   bash start_vlm_server.sh qwen35-122b              # Qwen/Qwen3.5-122B-A10B-FP8
#   bash start_vlm_server.sh internvl8b               # OpenGVLab/InternVL3_5-8B
#   bash start_vlm_server.sh internvl38b              # OpenGVLab/InternVL3_5-38B
#   bash start_vlm_server.sh cosmos                   # nvidia/Cosmos-Reason1-7B
#   bash start_vlm_server.sh cosmos2                  # nvidia/Cosmos-Reason2-8B
#   bash start_vlm_server.sh qwen3-8b                 # Qwen/Qwen3-8B (LLM judge)
#
# Environment:
#   GPU      GPU index (default: 0)
#   PORT     Server port (default: 8000)
#
# VRAM budget (192 GB B200):
#   Extractor (Qwen3-8B) runs alongside at 0.12 util (~23 GB).
#   Victim utilization + 0.12 ≤ 0.95 → victim budget ≤ 0.83.
#
#   ~7-9B BF16  : ~18 GB weights → 0.18 util (35 GB total)
#   ~27-33B BF16: ~54-66 GB weights → 0.45-0.50 util (~86-96 GB total)
#   ~38B BF16   : ~76 GB weights → 0.55 util (106 GB total)
#   ~122B FP8   : ~122 GB weights → 0.80 util (154 GB total)

set -euo pipefail

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
LOGDIR="./logs"
mkdir -p "$LOGDIR"

MODEL_KEY="${1:-qwen3vl8b}"

case "$MODEL_KEY" in
    # --- Small models (~7-9B, BF16, ~23 GB) ---
    qwen3vl8b|qwen3-vl-8b)
        MODEL="Qwen/Qwen3-VL-8B-Instruct"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS=""
        ;;
    qwen35-9b|qwen3.5-9b)
        MODEL="Qwen/Qwen3.5-9B"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS=""
        ;;
    internvl8b|internvl3.5-8b)
        MODEL="OpenGVLab/InternVL3_5-8B"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS="--trust-remote-code"
        ;;
    cosmos|cosmos-reason1-7b)
        MODEL="nvidia/Cosmos-Reason1-7B"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS="--trust-remote-code"
        ;;
    cosmos2|cosmos-reason2-8b)
        MODEL="nvidia/Cosmos-Reason2-8B"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS="--trust-remote-code"
        ;;
    qwen3-8b)
        MODEL="Qwen/Qwen3-8B"
        MAX_LEN=4096
        GPU_UTIL=0.18
        EXTRA_ARGS=""
        ;;

    # --- Medium models (~27-33B, BF16, ~65-78 GB) ---
    qwen35-27b|qwen3.5-27b)
        MODEL="Qwen/Qwen3.5-27B"
        MAX_LEN=4096
        GPU_UTIL=0.45
        EXTRA_ARGS=""
        ;;
    qwen3vl30b|qwen3-vl-30b)
        MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
        MAX_LEN=4096
        GPU_UTIL=0.50
        EXTRA_ARGS=""
        ;;
    qwen3vl32b|qwen3-vl-32b)
        MODEL="Qwen/Qwen3-VL-32B-Instruct"
        MAX_LEN=4096
        GPU_UTIL=0.50
        EXTRA_ARGS=""
        ;;

    # --- Large models (~38B, BF16, ~90 GB) ---
    internvl38b|internvl3.5-38b)
        MODEL="OpenGVLab/InternVL3_5-38B"
        MAX_LEN=4096
        GPU_UTIL=0.55
        EXTRA_ARGS="--trust-remote-code"
        ;;

    # --- XL models (~122B, FP8, ~143 GB) ---
    qwen35-122b|qwen3.5-122b)
        MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
        MAX_LEN=4096
        GPU_UTIL=0.80
        EXTRA_ARGS=""
        ;;

    *)
        echo "Unknown model key: $MODEL_KEY"
        echo "Supported: qwen3vl8b, qwen3vl30b, qwen3vl32b, qwen35-9b, qwen35-27b, qwen35-122b, internvl8b, internvl38b, cosmos, cosmos2, qwen3-8b"
        exit 1
        ;;
esac

# Kill existing server on the port
PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$PIDS" ]]; then
    echo "Killing existing processes on port $PORT: $PIDS"
    echo "$PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 3
fi

LOGFILE="$LOGDIR/vlm_server_${MODEL_KEY}.log"

echo "Starting VLM server:"
echo "  Model:    $MODEL"
echo "  GPU:      $GPU"
echo "  Port:     $PORT"
echo "  Max len:  $MAX_LEN"
echo "  GPU util: $GPU_UTIL"
echo "  Log:      $LOGFILE"

CUDA_VISIBLE_DEVICES="$GPU" nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    $EXTRA_ARGS \
    > "$LOGFILE" 2>&1 &
disown

# Wait for ready
echo -n "Waiting for server..."
ELAPSED=0
while ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [[ $ELAPSED -ge 600 ]]; then
        echo " TIMEOUT (${ELAPSED}s)"
        echo "Check log: $LOGFILE"
        exit 1
    fi
    echo -n "."
done
echo " ready (${ELAPSED}s)"
echo "Model: $(curl -s http://localhost:${PORT}/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
