#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 on two CMP 170HX cards, vLLM, PLE offloaded to host
# RAM in FP8 (see quantize_ple_fp8.py for why: the BF16 table is ~96 GiB, more
# than a 91 GB host can hold alongside everything else).
#
# Serves heretic-2 W4A16 by default (prepare-heretic2.sh); MODEL_DIR=$BASE_MODEL_DIR
# serves the Vnimanie W4A16 checkpoint the README's main tables were measured on.
#
# Differences from alesha-pro/qwen38-flash-next-4x3090, whose overlays this
# mounts: TP2 instead of TP4, a W4A16 checkpoint instead of their derived
# cyankiwi one, a locally derived FP8 PLE instead of RadixArk's, and BF16 KV
# instead of their calibrated FP8 QSA (their scales.json is calibrated for their
# composition, not this one; FP8 KV is still available, see KV_CACHE_DTYPE).
#
# VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75: the 5 s default let vLLM close idle keep-alive
# connections a LiteLLM gateway was about to reuse ("Server disconnected").
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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}   # the model's native maximum; a 200K-token needle was found (README)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}   # 4 concurrent streams: 333 tok/s aggregate vs 127 for one
# 8192 needs the smaller KV below: prefill activations grow with this, and an
# explicit --kv-cache-memory skips the profiling that would otherwise leave room
# for them. 1024 pairs with the old full-card KV of 24383208960.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
# auto = BF16. fp8_e4m3 needs QSA_SCALES (README, FP8 KV cache).
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
# Per card, in bytes. Measured with batched tokens 8192: 17.6 GiB here holds 8
# parallel 33K requests with 753 MiB to spare at peak; vLLM's own "fully
# utilize" suggestion of 19.76 GiB OOMs on an 8.9K prefill. Set KV_CACHE_MEMORY=
# (empty) to size by GPU_MEMORY_UTILIZATION instead (15.5 GiB, the safe side).
KV_CACHE_MEMORY=${KV_CACHE_MEMORY-18897856102}
# mode 0 = no inductor. VLLM_COMPILE was measured too: it starts (it does not
# hang, whatever older notes say) but decodes no faster and starts 30 s slower.
# FULL is not an option: the QSA attention backend only supports uniform
# batches, and vLLM downgrades FULL to FULL_DECODE_ONLY anyway.
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'}
# The checkpoint keeps its MTP draft head in BF16, so speculative decoding is
# available. k=4 measured best: code 173-182 tok/s vs 157 (k=3), 134 (k=2),
# 66 (off); prose is ~100-110 for any k. SPEC= (empty) disables it.
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

# FP8 KV and its calibration both go through alesha-pro's QSA overlays, whose
# base files hash-match this image. QSA_SCALES is a scales.json calibrated for
# THIS composition (results/fp8-kv-scales-w4a16-tp2-mtp4.json was, for the base
# checkpoint); QSA_CALIBRATE_OUT=<path> collects one instead.
QSA_MOUNTS=() QSA_ENV=()
if [[ $KV_CACHE_DTYPE == fp8* || -n ${QSA_CALIBRATE_OUT:-} ]]; then
    QSA_MOUNTS=(
        -v "$RECIPE/overlays/qwen38/qsa.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py:ro"
        -v "$RECIPE/overlays/qwen38/ops_qsa.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro"
    )
fi
if [[ $KV_CACHE_DTYPE == fp8* ]]; then
    [[ -f ${QSA_SCALES:-} ]] || { echo "FP8 KV needs QSA_SCALES=<scales.json>" >&2; exit 2; }
    QSA_MOUNTS+=(-v "$(realpath "$QSA_SCALES"):/qsa-scales/scales.json:ro")
    QSA_ENV+=(-e QSA_FP8_SCALES_FILE=/qsa-scales/scales.json)
fi
if [[ -n ${QSA_CALIBRATE_OUT:-} ]]; then
    mkdir -p "$(dirname "$QSA_CALIBRATE_OUT")"
    QSA_MOUNTS+=(-v "$(realpath "$(dirname "$QSA_CALIBRATE_OUT")"):/qsa-calib")
    QSA_ENV+=(-e "QSA_FP8_CALIBRATE_OUT=/qsa-calib/$(basename "$QSA_CALIBRATE_OUT")")
fi
# Extra vLLM arguments, word-split (e.g. EXTRA_ARGS="--enforce-eager").
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

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
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
    "${QSA_ENV[@]}" \
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
    "${QSA_MOUNTS[@]}" \
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
    "${EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"
