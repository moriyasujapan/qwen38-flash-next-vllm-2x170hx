# Shared settings for every script in this directory. Source it; override any
# value by exporting it first.

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Where the checkpoints live. Peak ~305 GB during preparation: 180 GB base W4A16
# + 74 GB heretic-2 + 51 GB derived FP8 PLE. Once prepare-heretic2.sh and
# build-ple-fp8.sh have run, the base checkpoint can be deleted (~125 GB left).
MODELS_DIR=${MODELS_DIR:-$REPO/models}

# The base W4A16 checkpoint, pinned to the revision loktar00's 3090 recipe
# measured. Served directly before heretic-2 became the default, and still the
# source of the FP8 PLE table, the BF16 MTP head and the tokenizer files.
W4A16_REPO=${W4A16_REPO:-VnimanieAI/Qwen3.8-Flash-Next-W4A16}
W4A16_REV=${W4A16_REV:-9236d703b25f25eb5c17e9640204f84fa1ce0c6e}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-W4A16}

# heretic-2 (trohrbaugh/Qwen3.8-Flash-Next-heretic-2, refusals removed) as AutoRound
# W4A16. Its own FP8 PLE files are not downloaded; the locally derived table is used.
HERETIC2_REPO=${HERETIC2_REPO:-densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE}
HERETIC2_REV=${HERETIC2_REV:-705a2ce7f529f61bcd94723bcde66866e0b63286}
HERETIC2_DIR=${HERETIC2_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-heretic2-W4A16}

# What run.sh serves. MODEL_DIR=$BASE_MODEL_DIR serves the base checkpoint instead.
MODEL_DIR=${MODEL_DIR:-$HERETIC2_DIR}
PLE_MODEL_DIR=${PLE_MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-PLE-FP8}

# The PLE offload overlays come from alesha-pro's 4x3090 recipe, pinned.
OVERLAY_REPO=${OVERLAY_REPO:-https://github.com/alesha-pro/qwen38-flash-next-4x3090.git}
OVERLAY_REV=${OVERLAY_REV:-9b36aaec3f604d726c6b751e29f7de8225fc203b}
RECIPE=${RECIPE:-$REPO/vendor/qwen38-flash-next-4x3090}

IMAGE=${IMAGE:-vllm/vllm-openai:qwen38-flash-next-x86_64-cu130}
