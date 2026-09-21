#!/usr/bin/env bash
# Fetch the pinned PLE-offload overlays and the vLLM image.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

if [[ ! -d $RECIPE/.git ]]; then
    git clone -q "$OVERLAY_REPO" "$RECIPE"
fi
git -C "$RECIPE" fetch -q origin
git -C "$RECIPE" checkout -q "$OVERLAY_REV"
echo "overlays: $RECIPE @ $(git -C "$RECIPE" rev-parse --short HEAD)"

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
echo "image: $IMAGE"
