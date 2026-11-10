#!/usr/bin/env bash
# Serve this checkpoint on the upstream vLLM wheel + vllm-cmp170hx plugin build
# (AlexBrandes) instead of the forked image scripts/run.sh uses. Measured faster on
# every axis; see "The upstream build is faster than this one" in the README.
#
# That build reads the PLE table from the checkpoint directory (upstream
# `--engram-config {"disk_offload":true}`), not from VLLM_PLE_MODEL_PATH, so it needs
# a composed model dir: the base checkpoint plus the FP8 PLE shards, with
# text_config.ple_embedding_dtype = "float8_e4m3fn" in config.json and the PLE
# tensors merged into model.safetensors.index.json. scripts/compose-upstream-model.sh
# builds one out of symlinks (22 MB, no copy).
#
#   ./scripts/compose-upstream-model.sh      # once
#   ./scripts/run-upstream.sh                # ~13 min to /health
#
# Rollback: docker stop flashnext-upstream && ./scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh 2>/dev/null || true

IMAGE=${IMAGE:-ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:a87f558b57e60b4f3c796d7be11d12199ce77f393df6ed1c40181ce932717757}
MODEL=${MODEL:-${ROOT:-/home/build/170hx/local-llm}/models/heretic2-upstream-ple}
NAME=${NAME:-flashnext-upstream}
PORT=${PORT:-18025}
GPUS=${GPUS:-GPU-979d3640-5aae-5d74-ed57-6b2db31a0e5e,GPU-39767737-ed80-6461-619c-61bf40a7573c}
TP=${TP:-2}
EP=${EP-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-12}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
SPEC=${SPEC-'{"method":"mtp","num_speculative_tokens":4}'}
ENGRAM=${ENGRAM-'{"disk_offload":true}'}
PLE_THREADS=${PLE_THREADS:-32}
EXTRA=(${EXTRA_ARGS:-})

EP_ARGS=(); [[ -n $EP ]] && EP_ARGS=(--enable-expert-parallel)
SPEC_ARGS=(); [[ -n $SPEC ]] && SPEC_ARGS=(--speculative-config "$SPEC")
ENGRAM_ARGS=(); [[ -n $ENGRAM ]] && ENGRAM_ARGS=(--engram-config "$ENGRAM")

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" --restart no \
    --gpus "\"device=$GPUS\"" \
    --ipc=host --shm-size 1g \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -p "127.0.0.1:$PORT:8000" \
    -e VLLM_PLE_DISK_READ_THREADS="$PLE_THREADS" \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
    -e HF_HOME=/hf-cache \
    -v "${CACHE:-${ROOT:-/home/build/170hx/local-llm}/models/.vllm-cache}:/hf-cache" \
    -v "$MODEL:/model:ro" \
    -v "${BASE_DIR:-${ROOT:-/home/build/170hx/local-llm}/models/Qwen3.8-Flash-Next-heretic2-W4A16-densohax}:/base:ro" \
    -v "${PLE_DIR:?set PLE_DIR to the FP8 PLE directory}:/ple:ro" \
    "$IMAGE" \
    vllm serve /model \
      --served-model-name flash-next-w4a16 \
      --tensor-parallel-size "$TP" \
      "${EP_ARGS[@]}" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --trust-remote-code \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      --no-enable-flashinfer-autotune \
      "${ENGRAM_ARGS[@]}" \
      "${SPEC_ARGS[@]}" \
      "${EXTRA[@]}" \
      --host 0.0.0.0 --port 8000
