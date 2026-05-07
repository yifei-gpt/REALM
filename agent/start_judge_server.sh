#!/usr/bin/env bash
# Start the Qwen3-8B LLM judge/extractor server for MCQ answer extraction.
#
# Usage:
#   bash start_judge_server.sh             # default: GPU 1, port 8002
#   GPU=0 PORT=8003 bash start_judge_server.sh
#
# Environment:
#   GPU      GPU index (default: 1)
#   PORT     Server port (default: 8002)

set -euo pipefail

GPU="${GPU:-1}"
PORT="${PORT:-8002}"
MODEL="Qwen/Qwen3-8B"
LOGDIR="./logs"
mkdir -p "$LOGDIR"

# Kill existing server on the port
PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$PIDS" ]]; then
    echo "Killing existing processes on port $PORT: $PIDS"
    echo "$PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 3
fi

LOGFILE="$LOGDIR/judge_server.log"

echo "Starting judge server:"
echo "  Model:  $MODEL"
echo "  GPU:    $GPU"
echo "  Port:   $PORT"
echo "  Log:    $LOGFILE"

CUDA_VISIBLE_DEVICES="$GPU" nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --enforce-eager \
    > "$LOGFILE" 2>&1 &
disown

# Wait for ready
echo -n "Waiting for server..."
ELAPSED=0
while ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [[ $ELAPSED -ge 300 ]]; then
        echo " TIMEOUT (${ELAPSED}s)"
        echo "Check log: $LOGFILE"
        exit 1
    fi
    echo -n "."
done
echo " ready (${ELAPSED}s)"
echo "Model: $(curl -s http://localhost:${PORT}/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
