#!/usr/bin/env bash
# Compose the model directory the upstream build needs, out of symlinks.
#
# Upstream's `--engram-config {"disk_offload":true}` keeps the PLE table in the
# checkpoint files and reads each step's rows from them, so it will not look at
# VLLM_PLE_MODEL_PATH the way the forked image does. This builds a directory that
# satisfies it without copying 122 GB:
#
#   - every base-checkpoint file, symlinked
#   - the FP8 PLE shards, symlinked (they already use upstream's shard_N names)
#   - a real config.json with text_config.ple_embedding_dtype = "float8_e4m3fn",
#     which is what selects Qwen4ExpPLEFp8EmbeddingMethod
#   - a real model.safetensors.index.json with the PLE tensors merged in
#
# The symlinks point at the *in-container* mounts (/base, /ple), so this directory
# only resolves inside the container run-upstream.sh starts. That is intentional.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh 2>/dev/null || true
ROOT=${ROOT:-/home/build/170hx/local-llm}
BASE_DIR=${BASE_DIR:-$ROOT/models/Qwen3.8-Flash-Next-heretic2-W4A16-densohax}
PLE_DIR=${PLE_DIR:?set PLE_DIR to the FP8 PLE directory (manifest.json + model-plefp8-*.safetensors)}
OUT=${OUT:-$ROOT/models/heretic2-upstream-ple}

[ -r "$BASE_DIR/config.json" ] || { echo "no checkpoint at $BASE_DIR" >&2; exit 1; }
[ -r "$PLE_DIR/manifest.json" ] || { echo "no FP8 PLE at $PLE_DIR" >&2; exit 1; }

mkdir -p "$OUT"
for f in "$BASE_DIR"/*; do ln -sfn "/base/$(basename "$f")" "$OUT/$(basename "$f")"; done
for f in "$PLE_DIR"/model-plefp8-*.safetensors "$PLE_DIR"/model-bf16-*.safetensors; do
  [ -e "$f" ] && ln -sfn "/ple/$(basename "$f")" "$OUT/$(basename "$f")"
done

BASE_DIR="$BASE_DIR" PLE_DIR="$PLE_DIR" OUT="$OUT" python3 - <<'PY'
import json, os, pathlib, struct
base, ple, out = (pathlib.Path(os.environ[k]) for k in ("BASE_DIR", "PLE_DIR", "OUT"))

cfg = json.loads((base / "config.json").read_text())
cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
p = out / "config.json"; p.unlink(missing_ok=True)
p.write_text(json.dumps(cfg, indent=1))

idx = json.loads((base / "model.safetensors.index.json").read_text())
wm, added = idx["weight_map"], 0
for f in sorted(ple.glob("model-plefp8-*.safetensors")) + sorted(ple.glob("model-bf16-*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    for k in hdr:
        wm[k] = f.name; added += 1
p = out / "model.safetensors.index.json"; p.unlink(missing_ok=True)
p.write_text(json.dumps(idx))
print(f"composed {out}: {added} PLE tensors merged, {len(wm)} tensors total")
PY
echo "now: PLE_DIR=$PLE_DIR ./scripts/run-upstream.sh"
