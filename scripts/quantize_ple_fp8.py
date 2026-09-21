#!/usr/bin/env python3
"""Build an external FP8 PLE checkpoint out of a BF16 PLE source.

The Flash-Next PLE n-gram table is ~96 GiB in BF16, which does not fit in this
host's 91 GB of RAM. alesha-pro's PLE offload overlay can read the table from a
second checkpoint in FP8 E4M3 with one global scale (~48 GiB), the layout
RadixArk/Qwen3.8-Flash-Next-NVFP4 ships. This writes that layout from the BF16
tensors already present in the W4A16 checkpoint, so no second 180 GB download.

Quantization matches the RadixArk composition: scale = amax / 448 over the whole
n-gram table (not per shard, not per row), applied to every shard. Everything
else under `.ple.` is copied through unchanged.

    python3 quantize_ple_fp8.py <src-checkpoint> <out-dir>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
SHARD_BYTES = 5 * 1024**3  # keep output files near RadixArk's ~5.2 GB


def ple_tensors(src: Path) -> dict[str, str]:
    index = json.loads((src / "model.safetensors.index.json").read_text())
    return {n: f for n, f in index["weight_map"].items() if ".ple." in n}


def is_ngram_shard(name: str) -> bool:
    return ".ngram_embedding.shard_" in name


def amax(src: Path, names: dict[str, str]) -> float:
    """Pass 1: the largest magnitude anywhere in the n-gram table."""
    peak = 0.0
    shards = sorted(n for n in names if is_ngram_shard(n))
    for i, name in enumerate(shards, 1):
        with safe_open(src / names[name], framework="pt") as f:
            t = f.get_tensor(name)
        peak = max(peak, t.abs().max().item())
        if i % 16 == 0 or i == len(shards):
            print(f"  amax pass {i}/{len(shards)}: {peak:.6g}", flush=True)
    return peak


def main() -> None:
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ple_tensors(src)
    shards = sorted((n for n in names if is_ngram_shard(n)),
                    key=lambda n: int(n.rsplit("shard_", 1)[1].split(".")[0]))
    others = [n for n in names if not is_ngram_shard(n)]
    print(f"{len(shards)} n-gram shards, {len(others)} other PLE tensors")

    peak = amax(src, names)
    scale = torch.tensor([peak / FP8_MAX], dtype=torch.bfloat16)
    # The scale is stored in BF16, so quantize against the value the loader will
    # actually see rather than the full-precision quotient.
    scale_f32 = scale.float().item()
    print(f"amax {peak:.10g} -> scale {scale_f32:.20g}", flush=True)

    weight_map: dict[str, str] = {}
    prefix = shards[0].rsplit(".ngram_embedding.", 1)[0] + ".ngram_embedding."

    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    file_no = 0

    def flush(kind: str) -> None:
        nonlocal buf, buf_bytes, file_no
        if not buf:
            return
        fn = f"model-{kind}-{file_no:05d}.safetensors"
        save_file(buf, str(out / fn))
        for n in buf:
            weight_map[n] = fn
        print(f"  wrote {fn} ({buf_bytes / 1e9:.2f} GB, {len(buf)} tensors)",
              flush=True)
        buf, buf_bytes, file_no = {}, 0, file_no + 1

    for i, name in enumerate(shards, 1):
        with safe_open(src / names[name], framework="pt") as f:
            t = f.get_tensor(name)
        q = (t.float() / scale_f32).clamp_(-FP8_MAX, FP8_MAX).to(FP8)
        del t
        buf[name] = q
        buf_bytes += q.numel()
        if buf_bytes >= SHARD_BYTES:
            flush("plefp8")
        if i % 16 == 0:
            print(f"  quantized {i}/{len(shards)}", flush=True)
    buf[prefix + "weight_scale"] = scale
    buf_bytes += 2
    flush("plefp8")

    file_no = 0
    for name in others:
        with safe_open(src / names[name], framework="pt") as f:
            buf[name] = f.get_tensor(name)
        buf_bytes += buf[name].numel() * buf[name].element_size()
        if buf_bytes >= SHARD_BYTES:
            flush("bf16")
    flush("bf16")

    total = sum((out / f).stat().st_size for f in set(weight_map.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}: {total / 1e9:.1f} GB across "
          f"{len(set(weight_map.values()))} files")


if __name__ == "__main__":
    main()
