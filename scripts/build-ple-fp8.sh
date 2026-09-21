#!/usr/bin/env bash
# Derive the FP8 external PLE checkpoint from the W4A16 checkpoint's BF16 PLE
# tensors, then write the manifest the offload overlay validates against.
#
# Runs inside the vLLM image because the conversion needs torch (FP8 dtypes) and
# safetensors. Reads ~96 GB, writes ~51 GB; a few minutes on NVMe.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "$PLE_MODEL_DIR"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$REPO/scripts:/scripts:ro" \
    -v "$MODEL_DIR:/src:ro" \
    -v "$PLE_MODEL_DIR:/out" \
    --entrypoint python3 \
    "$IMAGE" /scripts/quantize_ple_fp8.py /src /out

# The manifest tool reads a "revision" out of the HF cache layout; this
# checkpoint is derived locally, so stamp it with the source revision it came
# from rather than inventing a hash.
mkdir -p "$PLE_MODEL_DIR/.cache/huggingface/trees"
printf '{"derived_from": "%s@%s"}\n' "$W4A16_REPO" "$W4A16_REV" \
    > "$PLE_MODEL_DIR/.cache/huggingface/trees/$W4A16_REV.json"

python3 "$RECIPE/scripts/ple_manifest.py" generate "$PLE_MODEL_DIR" \
    "local/Qwen3.8-Flash-Next-PLE-FP8" "$PLE_MODEL_DIR/manifest.json" \
    '{"bf16_repo": "'"$W4A16_REPO"'", "bf16_revision": "'"$W4A16_REV"'", "method": "amax/448 global scale, scripts/quantize_ple_fp8.py"}'

python3 "$RECIPE/scripts/ple_manifest.py" validate "$PLE_MODEL_DIR" "$PLE_MODEL_DIR/manifest.json"
echo "PLE FP8 checkpoint ready: $PLE_MODEL_DIR"
