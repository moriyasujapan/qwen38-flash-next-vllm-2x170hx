# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[日本語](README_ja.md)

Serving **Qwen3.8-Flash-Next** (W4A16) on two **NVIDIA CMP 170HX** mining cards with vLLM,
on a host with **92 GiB of RAM** — less than the published recipes ask for.

**~100–115 tok/s on prose, ~175 tok/s on code, single stream · 333 tok/s aggregate at
4 concurrent · 1,058,505 KV tokens · 65K context · tool calling.**

Every number on this page was measured on this machine; how is in [Method](#method), and
the raw results are in `results/`.

---

## What is different here

The published Flash-Next W4A16 recipes
([loktar00](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe),
[alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090)) target 4× RTX 3090
and ask for 110–128 GB of host RAM. Two things make it fit on two 170HX and 92 GiB:

1. **The PLE n-gram table is converted to FP8 locally.** The W4A16 checkpoint keeps the
   51B-parameter per-layer-embedding n-gram table in BF16 (102.4 GB / 95.4 GiB), and
   vLLM's PLE offload holds it in host RAM. `scripts/quantize_ple_fp8.py` rewrites it as
   FP8 E4M3 with one global scale (`amax / 448`) — the layout alesha-pro's offload overlay
   reads — which brings it to 51.3 GB / 47.7 GiB. The derived scale,
   `0.00019931793212890625`, is **bit-identical** to the one in RadixArk's published FP8
   PLE: the same table, without a second 180 GB download.
2. **TP2 + expert parallel on 64 GB cards.** Expert parallel is required: the MoE
   intermediate size 640 split two ways is 320, which the quant's group size of 128 does
   not divide.

## Hardware, measured

| | |
|---|---|
| GPUs | 2× CMP 170HX: GA100, sm_80, 70 SMs, 63.4 GiB usable each |
| Interconnect | PCIe Gen2 x16 per card (`lspci` LnkSta 5 GT/s x16), **no P2P** (`can_device_access_peer` false both ways) |
| Host ↔ device | **6.6 GB/s** (pinned, 256 MB copies), ~83% of the Gen2 x16 ceiling |
| Host RAM | 91.9 GiB |

Some 170HX write-ups describe the link as fused down to x4. These cards are not; check
yours with `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current
--format=csv` and a copy benchmark, since an x4 card moves a quarter of the data.

## Results

Default configuration: MTP k=4, `--max-num-seqs 8`, `--max-num-batched-tokens 1024`,
BF16 KV, `--kv-cache-memory` sized to fill the card.

### Decode, single stream (tok/s, median of 3, 512 max tokens)

| config | ja prose | en prose | ja list | code |
|---|---|---|---|---|
| no MTP | 65.8 | 65.9 | 64.9 | 66.5 |
| MTP k=2 | 105.3 | 102.2 | 101.2 | 133.6 |
| MTP k=3 | 105.9 | 104.6 | 111.1 | 156.5 |
| MTP k=4, `max-num-seqs 1` | 99.6 | 100.9 | 104.9 | 182.0 |
| **MTP k=4, `max-num-seqs 8` (default)** | **104.5** | **101.1** | **114.0** | **173.0** |

MTP is worth ~1.6× on prose and ~2.6× on code. Larger k helps code (predictable tokens)
and makes no difference to prose. The min–max spread within a cell is typically ±5–10%.

### Concurrency (aggregate tok/s, 256-token code answers)

| concurrent requests | 1 | 4 | 8 |
|---|---|---|---|
| `max-num-seqs 1` | 127 | 126 (queued) | – |
| **`max-num-seqs 8`** | 127 | **333** | 302 |

### Prefill (prompt tok/s, prefix cache defeated with a random nonce)

| prompt | 6,954 tokens | 27,853 tokens |
|---|---|---|
| default | 2,637 | 3,068 |

### Things that did not help

| tried | result |
|---|---|
| `--max-num-batched-tokens 4096` | HTTP 500 on a 7K-token prefill; decode unchanged |
| `VLLM_COMPILE` (inductor) | starts fine — it does **not** hang, despite older notes — but decodes no faster (124 / 101 / 112 / 176) and starts 30 s slower |
| `--pipeline-parallel-size 2` | refused at startup: `VLLM_PLE_CPU_OFFLOAD does not support the requested configuration. Unsupported settings: PP=2` |
| `cudagraph_mode: FULL` | downgraded by vLLM to `FULL_DECODE_ONLY` (the QSA backend supports uniform batches only) |
| FP8 KV (`fp8_e4m3`, alesha-pro's QSA overlays, scales calibrated for this composition) | works, but not worth it here — see below |

### FP8 KV cache

Calibrated with the overlay's own collector (`QSA_FP8_CALIBRATE_OUT`) on this checkpoint,
TP2 and MTP k=4, taking per layer the larger of this calibration and alesha-pro's; the
result is in `results/fp8-kv-scales-w4a16-tp2-mtp4.json`. The MTP drafter's attention
layer has no entry in alesha-pro's file, so calibrating here was necessary.

| | BF16 KV (default) | FP8 KV |
|---|---|---|
| KV tokens | 1,058,505 | 1,441,792 (+36%; the linear-attention state stays BF16) |
| decode ja / en / list / code | 104.5 / 101.1 / 114.0 / 173.0 | 95.5 / 94.9 / 101.2 / 158.0 (−6 to −11%) |
| prefill 7K / 28K | 2,637 / 3,068 | 2,668 / 2,603 |
| needle, 29K-43K tokens, chat format, 12 runs | 11 found, 1 refused | 10 found, 2 refused |
| 16 short greedy prompts, 64 tokens | – | 12 identical to BF16; 4 reworded, none broken |

Retrieval matched BF16 (no run failed to find the needle; the misses were the model
declining to reveal a "vault override code"). But decode is slower and the extra capacity
is not needed: BF16 already holds eight concurrent 65K contexts twice over. Default stays BF16.

### Long-form output

The 5-section Japanese report prompt in `bench/report-prompt-ja.txt` produced 6,813
characters / 4,080 tokens in 44 s with `reasoning_effort: medium`, finished cleanly, every
section present, no garbled Japanese. It also confidently expanded an acronym it did not
know and filled a comparison table with unsourced numbers — verify specifics.

### Where the memory goes

| | per card | total |
|---|---|---|
| weights + non-torch (incl. MTP head) | 38.7 GiB | 77.4 GiB |
| KV cache (BF16) | 22.7 GiB | 45.4 GiB → 1,058,505 tokens |
| host RAM, whole container | – | 61.2 GiB (PLE worker 48.4 GiB) |

Startup takes about **5 min 50 s**: main weights 103 s, MTP drafter 9 s, FP8 PLE ~40 s,
engine init and CUDA graph capture 122 s.

## Use `reasoning_effort: medium` (or low). Never leave it at the default.

The chat template defaults to `xhigh`. Measured on three coding/reasoning prompts, two runs
per level, max_tokens 20,000 (reasoning tokens per run):

| prompt | low | medium | xhigh |
|---|---|---|---|
| CSV statistics function | 176 / 205 | 196 / 117 | 1,114 / **20,000, truncated** |
| asyncio server slowdown | 238 / 307 | 440 / 335 | **16,577 / 19,451, both truncated** |
| algorithm with proof | 557 / 693 | 1,302 / 451 | **19,403 / 20,000, both truncated** |

low and medium finished every answer. **xhigh spent the whole budget thinking and never
answered in 5 of 6 runs** — about three minutes of nothing each. Send
`chat_template_kwargs: {"reasoning_effort": "medium"}` (valid values: `xhigh`, `medium`,
`low`; `high` is rejected). Clients that do not send it get `xhigh`.

## Requirements

- **Two 64 GB sm_80 cards.** Only CMP 170HX has been tested.
- **~64 GiB of free host RAM** for the container (61.2 GiB measured). On this 92 GiB host
  that leaves ~31 GiB.
- **~260 GB of disk**: 179.8 GB checkpoint + 51.3 GB derived PLE + 28.8 GB image.
- Docker with the NVIDIA runtime and a driver new enough for the CUDA 13.0 image.
- If a 170HX runs far below these numbers, check the motherboard `PWRBRK#` power brake
  first — see
  [deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx#troubleshooting-cards-running-4-slow-pwrbrk--edge-pin-b30).
  (Here `HW Power Brake Slowdown` reads `Not Active` on both cards.)

## Quick start

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx
cd qwen38-flash-next-vllm-2x170hx

./scripts/setup.sh            # pinned overlays + vLLM image (28.8 GB)
./scripts/download-model.sh   # 179.8 GB, rate-limited (RATE=95M), resumable
./scripts/build-ple-fp8.sh    # BF16 PLE -> FP8 + manifest

GPUS=1,2 ./scripts/run.sh     # the two 170HX, by index or UUID (nvidia-smi -L)
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py --reps 3 --prefill 8192,32768 --conc 1,4,8
```

The endpoint is `http://localhost:18024/v1`, model `flash-next-w4a16`, with the
`qwen3_xml` tool-call parser and `qwen3` reasoning parser. `gateway/litellm-config.yaml` is
an optional LiteLLM front end. Every path and setting is an environment variable; see
`scripts/env.sh` and the top of `scripts/run.sh`.

## Settings worth knowing

| setting | value | why |
|---|---|---|
| `--tensor-parallel-size 2 --enable-expert-parallel` | required | 640/2 is not divisible by the group size 128 |
| `--speculative-config` | MTP, k=4 | fastest on code, equal on prose (table above) |
| `--max-num-seqs` | 8 | 2.6× aggregate at 4 concurrent; single stream unchanged |
| `--max-num-batched-tokens` | 1024 | 4096 returned HTTP 500 on a 7K prefill |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | inductor gains nothing here; FULL is downgraded |
| `--kv-cache-memory` | 24383208960 | fills a 64 GB card; the fraction-based default left 4.35 GiB per card idle |
| `--kv-cache-dtype` | `auto` (BF16) | FP8 KV measured: +36% capacity, 6-11% slower decode |
| `VLLM_PLE_EMBEDDING_DTYPE` | `float8_e4m3fn` | selects the FP8 PLE path in the overlay |
| `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` | kept from the upstream recipe | the PLE offload worker uses `pidfd_getfd`; not re-tested without it |

## Method

- `bench/bench.py`: streaming requests, `temperature 0.7`, `reasoning_effort: medium`.
  Decode tok/s = (completion tokens − 1) / (last token − first token), so TTFT is
  excluded; completion tokens include reasoning. Prefill tok/s = prompt tokens / TTFT, with
  a random nonce at the start of every prompt so the prefix cache cannot serve it.
- Each configuration ran in a fresh container with nothing else sent to the server, except
  the run labelled `E` in `results/`, whose decode numbers overlapped with other requests
  and are not used above; the default row is its clean re-run, `FINAL`.
- The `batched-tokens 4096` run failed before it could write its result line.
- Needle tests go through the chat template. Sent as raw `/v1/completions` text, the same
  documents made the model stop after one token on BF16 and FP8 alike, depending only on
  how the document began; an early FP8 "failure" turned out to be this, not the cache.

## Credits

- Qwen for the model; [VnimanieAI](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
  for the W4A16 quantization.
- [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  for the PLE offload overlays and manifest tooling this mounts (pinned, not vendored).
- [loktar00/qwen38-flash-next-vllm-3090-recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)
  for the starting settings and MTP findings.
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  for the FP8 PLE layout this reproduces.
- vLLM.

## License

Apache-2.0 for the code in this repository. No weights are included; those are under the
Qwen Community License.
