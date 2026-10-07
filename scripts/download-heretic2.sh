#!/usr/bin/env bash
# Resumable, rate-limited download of the heretic-2 W4A16 checkpoint (~74 GB),
# sha256-verified. Only the top-level files: the repo's own FP8 PLE shards
# (model-plefp8-*) and its runtime/ directory are skipped, because run.sh uses
# the PLE table derived by build-ple-fp8.sh and the MTP head grafted by
# prepare-heretic2.sh.
#
# 5 streams x RATE each (default 18M, ~94 MB/s total); rerunning resumes.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RATE=${RATE:-18M}
STREAMS=${STREAMS:-5}

mkdir -p "$HERETIC2_DIR"
cd "$HERETIC2_DIR" || exit 2
echo "$HERETIC2_REV" > REVISION
curl -sf "https://huggingface.co/api/models/$HERETIC2_REPO/tree/$HERETIC2_REV?recursive=true" |
    python3 -c '
import json, sys
for x in json.load(sys.stdin):
    p = x["path"]
    if x["type"] != "file" or "/" in p or p == ".gitattributes" or p.startswith("model-plefp8-"):
        continue
    print(p, x.get("size", 0), (x.get("lfs") or {}).get("oid", ""))
' > files.txt || { echo "could not list $HERETIC2_REPO@$HERETIC2_REV" >&2; exit 2; }

dl() {
    f=$1 size=$2 sha=$3
    for _ in 1 2 3 4 5 6; do
        [[ -f $f && $(stat -c %s "$f") == "$size" ]] && break
        curl -sSL --fail --retry 10 --retry-all-errors -C - --limit-rate "$RATE" -o "$f" "$U/$f"
    done
    [[ $(stat -c %s "$f" 2>/dev/null) == "$size" ]] || { echo "FAIL size $f"; return; }
    [[ -z $sha || $(sha256sum "$f" | cut -d' ' -f1) == "$sha" ]] || { echo "FAIL sha $f"; return; }
    echo "OK $f"
}
export -f dl
export RATE U="https://huggingface.co/$HERETIC2_REPO/resolve/$HERETIC2_REV"
xargs -P "$STREAMS" -L 1 bash -c 'dl "$0" "$1" "$2"' < files.txt | tee fetch.log | grep FAIL
echo "done: $(grep -c '^OK' fetch.log)/$(wc -l < files.txt) ok"
! grep -q FAIL fetch.log
