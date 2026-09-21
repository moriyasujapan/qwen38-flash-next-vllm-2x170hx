#!/usr/bin/env bash
# Resumable, rate-limited download of the W4A16 checkpoint (~180 GB).
#
# curl rather than `hf download`: the latter has no rate limit and saturates a
# gigabit line. RATE caps it; rerunning resumes partial files.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RATE=${RATE:-95M}

mkdir -p "$MODEL_DIR/.cache/huggingface/trees"
echo '{}' > "$MODEL_DIR/.cache/huggingface/trees/$W4A16_REV.json"
curl -sf "https://huggingface.co/api/models/$W4A16_REPO/tree/$W4A16_REV" |
    python3 -c 'import json,sys;[print(x["path"],x.get("size",0)) for x in json.load(sys.stdin) if x["type"]=="file"]' \
    > "$MODEL_DIR/.filelist"

fail=0
while read -r f size; do
    out=$MODEL_DIR/$f
    [[ -f $out && $(stat -c %s "$out") == "$size" ]] && continue
    for _ in 1 2 3 4 5; do
        curl -sfL --limit-rate "$RATE" -C - -o "$out" \
            "https://huggingface.co/$W4A16_REPO/resolve/$W4A16_REV/$f" && break
        sleep 5
    done
    if [[ $(stat -c %s "$out" 2>/dev/null) == "$size" ]]; then
        echo "OK $f"
    else
        echo "FAIL $f"; fail=1
    fi
done < "$MODEL_DIR/.filelist"
exit $fail
