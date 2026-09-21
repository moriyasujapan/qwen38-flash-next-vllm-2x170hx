# Shared settings for every script in this directory. Source it; override any
# value by exporting it first.

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Where the checkpoints live. ~220 GB: 168 GB W4A16 + 48 GB derived FP8 PLE.
MODELS_DIR=${MODELS_DIR:-$REPO/models}

# The W4A16 checkpoint, pinned to the revision loktar00's 3090 recipe measured.
W4A16_REPO=${W4A16_REPO:-VnimanieAI/Qwen3.8-Flash-Next-W4A16}
W4A16_REV=${W4A16_REV:-9236d703b25f25eb5c17e9640204f84fa1ce0c6e}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-W4A16}
PLE_MODEL_DIR=${PLE_MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-PLE-FP8}

# The PLE offload overlays come from alesha-pro's 4x3090 recipe, pinned.
OVERLAY_REPO=${OVERLAY_REPO:-https://github.com/alesha-pro/qwen38-flash-next-4x3090.git}
OVERLAY_REV=${OVERLAY_REV:-9b36aaec3f604d726c6b751e29f7de8225fc203b}
RECIPE=${RECIPE:-$REPO/vendor/qwen38-flash-next-4x3090}

IMAGE=${IMAGE:-vllm/vllm-openai:qwen38-flash-next-x86_64-cu130}
