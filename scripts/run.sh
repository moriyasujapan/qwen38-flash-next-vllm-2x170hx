#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 on two CMP 170HX cards, vLLM, PLE offloaded to host
# RAM in FP8 (see quantize_ple_fp8.py for why: the BF16 table is ~96 GiB, more
# than a 91 GB host can hold alongside everything else).
#
# Differences from alesha-pro/qwen38-flash-next-4x3090, whose overlays this
# mounts: TP2 instead of TP4, the Vnimanie W4A16 checkpoint instead of their
# derived cyankiwi one, a locally derived FP8 PLE instead of RadixArk's, and
# BF16 KV instead of their calibrated FP8 QSA (their scales.json is calibrated
# for their composition, not this one).
#
# Expert parallel is not optional: moe_intermediate_size 640 sharded by TP2 is
# 320, which the group size of 128 does not divide.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

PLE_MANIFEST=${PLE_MANIFEST:-$PLE_MODEL_DIR/manifest.json}
CACHE=${CACHE:-$MODELS_DIR/.vllm-cache}
NAME=${NAME:-flashnext-vllm}
PORT=${PORT:-18024}
# The two 64 GB cards, by index or UUID (`nvidia-smi -L`). Keep smaller cards out
# of the group: TP caps every rank at the smallest card.
GPUS=${GPUS:?set GPUS to the two CMP 170HX devices, e.g. GPUS=1,2 or their UUIDs}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
# Sizing KV by fraction leaves ~4.5 GiB per card unused, because the profiler
# measures against free memory rather than the utilization target. vLLM prints
# the exact byte count that fills the card; this is it, per 64 GB card with MTP
# on. Set KV_CACHE_MEMORY= (empty) to fall back to the fraction.
KV_CACHE_MEMORY=${KV_CACHE_MEMORY:-24383208960}
# mode 0 = no inductor. Inductor compilation of this architecture hangs on
# Ampere; decode CUDA graphs alone are the difference between 9 and 105 tok/s.
# FULL is not an option even with MTP: the QSA attention backend only supports
# uniform batches, and vLLM downgrades FULL to FULL_DECODE_ONLY anyway.
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'}
# The checkpoint keeps its MTP draft head in BF16, so speculative decoding is
# available and worth +60% single-stream. SPEC= (empty) disables it.
SPEC=${SPEC-'{"method":"mtp","num_speculative_tokens":4}'}
DIST=/usr/local/lib/python3.12/dist-packages
VLLM_PKG=$DIST/vllm

for f in "$MODEL_DIR/model.safetensors.index.json" \
         "$PLE_MODEL_DIR/model.safetensors.index.json" "$PLE_MANIFEST"; do
    [[ -f $f ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$CACHE/vllm" "$CACHE/torchinductor"
docker rm -f "$NAME" 2>/dev/null || true

SPEC_ARGS=()
[[ -n $SPEC ]] && SPEC_ARGS=(--speculative-config "$SPEC")
KV_ARGS=()
[[ -n $KV_CACHE_MEMORY ]] && KV_ARGS=(--kv-cache-memory "$KV_CACHE_MEMORY")

exec docker run -d --name "$NAME" --restart unless-stopped \
    --gpus "\"device=$GPUS\"" \
    --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -p "$PORT:$PORT" \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_PLE_MODEL_PATH=/ple-model \
    -e VLLM_PLE_MANIFEST=/ple-manifest/manifest.json \
    -e VLLM_PLE_EMBEDDING_DTYPE=float8_e4m3fn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/hf-cache \
    -e TORCHINDUCTOR_CACHE_DIR=/compile-cache/torchinductor \
    -v "$MODEL_DIR:/model:ro" \
    -v "$PLE_MODEL_DIR:/ple-model:ro" \
    -v "$PLE_MANIFEST:/ple-manifest/manifest.json:ro" \
    -v "$CACHE:/hf-cache" \
    -v "$CACHE/vllm:/root/.cache/vllm" \
    -v "$CACHE/torchinductor:/compile-cache/torchinductor" \
    -v "$RECIPE/overlays/ple_layer.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" \
    -v "$RECIPE/overlays/ple_offload/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro" \
    -v "$RECIPE/overlays/ple_offload/ple_external_source.py:$VLLM_PKG/v1/ple_offload/ple_external_source.py:ro" \
    -v "$RECIPE/overlays/gpu_worker.py:$VLLM_PKG/v1/worker/gpu_worker.py:ro" \
    -v "$RECIPE/overlays/multiproc_executor.py:$VLLM_PKG/v1/executor/multiproc_executor.py:ro" \
    -v "$RECIPE/overlays/qwen_gdn_linear_attn.py:$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro" \
    -v "$RECIPE/overlays/shared_experts.py:$VLLM_PKG/model_executor/layers/fused_moe/runner/shared_experts.py:ro" \
    "$IMAGE" \
    /model \
    --served-model-name flash-next-w4a16 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --compilation-config "$COMPILATION_CONFIG" \
    "${SPEC_ARGS[@]}" \
    "${KV_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --no-enable-flashinfer-autotune \
    --host 0.0.0.0 \
    --port "$PORT"
