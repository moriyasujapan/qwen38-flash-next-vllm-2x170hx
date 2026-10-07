#!/usr/bin/env bash
# Rewrite the downloaded heretic-2 checkpoint so the pinned image serves it with
# MTP (see prepare_heretic2.py). Needs the base W4A16 checkpoint for its config,
# tokenizer and BF16 MTP head; writes a 4.9 GiB graft file. Runs inside the vLLM
# image for torch + safetensors.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

for f in "$HERETIC2_DIR/config.json" "$BASE_MODEL_DIR/model.safetensors.index.json"; do
    [[ -f $f ]] || { echo "missing $f" >&2; exit 2; }
done
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$REPO/scripts:/scripts:ro" \
    -v "$HERETIC2_DIR:/dst" \
    -v "$BASE_MODEL_DIR:/base:ro" \
    --entrypoint python3 \
    "$IMAGE" /scripts/prepare_heretic2.py /dst /base
