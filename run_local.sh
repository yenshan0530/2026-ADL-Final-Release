#!/usr/bin/env bash
# Start a vLLM server with the course model, play, then stop the server.
#
#   bash run_local.sh
#
# 1. Uncomment ONE of the two runs at the bottom (an economy or a market).
# 2. Activate the environment first (uv sync && source .venv/bin/activate).
# Every setting of the run is in its config file. Server settings, if you need others:
#   GPU=1 bash run_local.sh          another GPU
#   GPU_MEM=0.6 bash run_local.sh    a smaller share of the GPU's memory (if it is shared)
#   PORT=8020 bash run_local.sh      another port; then also set it in configs/models.yaml
#                                    (base_url of `base` and `ta`, and serving.base.port)
# The server matches `serving` in configs/models.yaml (as --launch-vllm starts it); your
# LoRAs are loaded into it when the run starts. To keep a server up across many runs, start
# the `vllm serve` line below in another terminal and run the scripts yourself.
# One Qwen3.5-4B server fits a 24 GB GPU; the first start downloads the model (~9 GB) and
# takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEM:-0.85}

# ---- start vLLM (in its own process group, so stopping it stops all its processes) ----
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  setsid vllm serve Qwen/Qwen3.5-4B --host localhost --port $PORT --enable-lora \
    --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64 --gpu-memory-utilization $GPU_MEM \
    --max-model-len 64000 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --language-model-only --reasoning-parser qwen3 --max-num-seqs 64 \
    > logs/vllm.log 2>&1 &
VLLM_PID=$!

stop_vllm() {                 # stop every process of the vLLM group (server and engine)
  kill -0 -- -$VLLM_PID 2>/dev/null || return 0
  echo "stopping vLLM ..."
  kill -TERM -- -$VLLM_PID 2>/dev/null || true
  for _ in $(seq 60); do                                  # up to 60 s, then force it
    kill -0 -- -$VLLM_PID 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- -$VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
  echo "vLLM stopped"
}
trap stop_vllm EXIT           # also when the run fails ...
trap 'exit 130' INT TERM      # ... or is stopped with Ctrl-C / kill

echo "starting vLLM (log: logs/vllm.log) ..."
until curl -sf http://localhost:$PORT/health > /dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM stopped; see logs/vllm.log" >&2; exit 1; }
  sleep 5
done
echo "vLLM is up"

# ---- play: uncomment one ------------------------------------------------------------
# To run an economy, uncomment this (--phase 1: student sellers, 2: student buyers, 3: both):
# python run_economy.py --config configs/economy.yaml --phase 2

# To run a market (exactly the agents listed in the config), uncomment this:
# python run_market.py --config configs/market.yaml

# ---- stop vLLM ------------------------------------------------------------------------
stop_vllm
