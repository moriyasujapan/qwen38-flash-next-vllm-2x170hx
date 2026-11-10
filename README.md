# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[日本語](README_ja.md)

Serving **Qwen3.8-Flash-Next** (W4A16) on two **NVIDIA CMP 170HX** mining cards with vLLM,
on a host with **92 GiB of RAM** — less than the published recipes ask for.

**~100–110 tok/s on prose, ~170 tok/s on code, single stream · ~390 tok/s aggregate at
4 concurrent · 262,144-token context (the model's native maximum; a 200K-token needle is
found) · tool calling.**

Every number on this page was measured on this machine; how is in [Method](#method), and
the raw results are in `results/`.

> **The default model is now heretic-2** (refusals removed), not the Vnimanie W4A16
> checkpoint the tables under [Results](#results) were measured on, and the defaults moved
> to `--max-num-batched-tokens 8192` with a 17.6 GiB-per-card KV cache. See
> [Model: heretic-2](#model-heretic-2) and [Batched tokens and KV size](#batched-tokens-and-kv-size).
> `MODEL_DIR=$BASE_MODEL_DIR` still serves the Vnimanie checkpoint.

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

## Model: heretic-2

`run.sh` serves [densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE](https://huggingface.co/densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE)
(revision `705a2ce`), an AutoRound W4A16 quant of
[trohrbaugh/Qwen3.8-Flash-Next-heretic-2](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2),
a version of the model with its refusals removed. It is the same architecture, so it runs
on the same image and overlays, after `scripts/prepare-heretic2.sh` rewrites it:

- **Quantization config.** The image has no `auto-round` method, but its AutoGPTQ path reads
  the same packing. `config.json` becomes the base checkpoint's with a GPTQ w4 g128 sym
  block; only routed experts are quantized (73,728 `qweight` tensors, all under `.experts.`),
  and `dynamic: {"-:.*mtp.*": {}}` keeps the MTP head BF16. The routed experts run on
  Marlin (`Using MarlinExperts` in the log), whatever `--quantization` says: the image maps
  `gptq`, `gptq_marlin` and `marlin` all to `auto_gptq`.
- **MTP head.** The checkpoint ships none, so the base checkpoint's BF16 MTP head is copied in
  as `model-mtp-graft.safetensors` (4.9 GiB). It was trained against the unmodified model;
  verification keeps the target's tokens either way.
- **Tokenizer, chat template, generation config.** Taken from the base checkpoint (the
  tokenizer files are byte-identical to it after the rewrite); the heretic
  `generation_config.json` stops on token 248044 only, the base one on 248046 and 248044.
- **PLE.** The FP8 table derived from the base checkpoint (`build-ple-fp8.sh`) is used; the
  repo's own FP8 PLE shards are not downloaded.

Measured on heretic-2 with the previous defaults (batched tokens 1024, 22.7 GiB KV per card),
in a [depth-ladder report on bench-of-us](https://github.com/jimoto-no-llm/bench-of-us/blob/main/report/2026-09-27_153558_measuring_qwen3.8_flash_next_w4a16_on_2x_cmp_170hx_with_vllm_tp2.md):

| depth | prefill (new tokens/s) | decode, synthetic text | decode, real-style generation |
|---|---|---|---|
| 32K | 3,323 | 186.2 (MTP acceptance 0.94) | 97.1 (0.39) |
| 128K | 2,994 | 184.6 (0.95) | 89.1 (0.35) |
| 258K | 2,575 | 178.3 (0.93) | 69.7 (0.21) |

The 2.6× gap between the two decode columns is MTP acceptance alone: a synthetic ladder
overstates decode for an MTP model. The heretic-2 figures are not directly comparable with
the [Results](#results) tables below, which use `bench/bench.py` on the base checkpoint.

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

Measured on the **base Vnimanie checkpoint** with the configuration that was the default at
the time: `--max-model-len 262144`, MTP k=4, `--max-num-seqs 8`,
`--max-num-batched-tokens 1024`, BF16 KV, `--kv-cache-memory 24383208960` (fills the card).
Rows marked 64K were measured with `--max-model-len 65536`, before that default moved. The
current defaults are in [Batched tokens and KV size](#batched-tokens-and-kv-size).

### Decode, single stream (tok/s, median of 3, 512 max tokens)

| config | ja prose | en prose | ja list | code |
|---|---|---|---|---|
| no MTP | 65.8 | 65.9 | 64.9 | 66.5 |
| MTP k=2 | 105.3 | 102.2 | 101.2 | 133.6 |
| MTP k=3 | 105.9 | 104.6 | 111.1 | 156.5 |
| MTP k=4, `max-num-seqs 1` (64K) | 99.6 | 100.9 | 104.9 | 182.0 |
| MTP k=4, `max-num-seqs 8` (64K) | 104.5 | 101.1 | 114.0 | 173.0 |
| **MTP k=4, `max-num-seqs 8`, 262K context (default)** | **101.3** | **97.8** | **107.7** | **170.0** |

MTP is worth ~1.6× on prose and ~2.6× on code. Larger k helps code (predictable tokens)
and makes no difference to prose. The min–max spread within a cell is typically ±5–10%.

### Concurrency (aggregate tok/s, 256-token code answers)

| concurrent requests | 1 | 4 | 8 |
|---|---|---|---|
| `max-num-seqs 1` (64K) | 127 | 126 (queued) | – |
| `max-num-seqs 8` (64K) | 127 | 333 | 302 |
| **`max-num-seqs 8`, 262K context (default)** | 126 | **392** | 387 |

### Prefill (prompt tok/s, prefix cache defeated with a random nonce)

| prompt | 6,954 tokens | 27,853 tokens |
|---|---|---|
| 64K | 2,637 | 3,068 |
| 262K context (default) | 3,191 | 3,261 |

### Long context

A 200,087-token chat prompt with a random code planted at 24% depth: the code was
returned correctly, time to first token 70.5 s (2,839 tok/s prefill), then 69.0 tok/s
decode at that depth. A repeated prefix is served from the prefix cache, so only the
first pass over a long document pays the prefill.

### Things that did not help

| tried | result |
|---|---|
| `--max-num-batched-tokens 4096` with the full-card KV (24383208960) | HTTP 500 on a 7K-token prefill; decode unchanged. With a smaller KV, 8192 works — see below |
| `VLLM_COMPILE` (inductor) | starts fine — it does **not** hang, despite older notes — but decodes no faster (124 / 101 / 112 / 176) and starts 30 s slower |
| `--pipeline-parallel-size 2` | refused at startup: `VLLM_PLE_CPU_OFFLOAD does not support the requested configuration. Unsupported settings: PP=2`. It does work on the upstream build — see [The upstream build is faster than this one](#the-upstream-build-is-faster-than-this-one) |
| `cudagraph_mode: FULL` | downgraded by vLLM to `FULL_DECODE_ONLY` (the QSA backend supports uniform batches only) |
| FP8 KV (`fp8_e4m3`, alesha-pro's QSA overlays, scales calibrated for this composition) | works, but not worth it here — see below |

### Batched tokens and KV size

`--max-num-batched-tokens 8192` (8× the old 1024) needs room for prefill activations that
the full-card KV left no space for, and an explicit `--kv-cache-memory` skips the profiling
that would otherwise reserve it. Measured with 8192, heretic-2, per card:

| `--kv-cache-memory` | result |
|---|---|
| 21213949952 (19.76 GiB), vLLM's own "to fully utilize gpu memory" suggestion | runtime OOM on an 8.9K-token prefill |
| **18897856102 (17.6 GiB), default** | 1,151,760 KV tokens, 4.39× concurrency at 262K. 8 parallel 33K-token requests pass; peak 64,783 / 65,536 MiB, **753 MiB to spare** |
| empty (sized by `--gpu-memory-utilization 0.92`, 15.5 GiB) | 1,016,505 KV tokens; the safe side |

The margin at 17.6 GiB is thin. If it OOMs, `KV_CACHE_MEMORY= ./scripts/run.sh` falls back
to the fraction, and `MAX_NUM_BATCHED_TOKENS=1024 KV_CACHE_MEMORY=24383208960` restores the
configuration the Results tables were measured with.

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
is not needed: BF16 already holds 5.67 full 262K contexts. Default stays BF16. (The FP8 run
used `--max-model-len 65536`.)

### Long-form output

The 5-section Japanese report prompt in `bench/report-prompt-ja.txt` produced 6,813
characters / 4,080 tokens in 44 s with `reasoning_effort: medium`, finished cleanly, every
section present, no garbled Japanese. It also confidently expanded an acronym it did not
know and filled a comparison table with unsourced numbers — verify specifics.

### Where the memory goes

| | per card | total |
|---|---|---|
| weights + non-torch (incl. MTP head) | 38.7 GiB | 77.4 GiB |
| KV cache (BF16), as measured | 22.7 GiB | 45.4 GiB → 1,486,412 tokens at 262K context |
| KV cache (BF16), current default | 17.6 GiB | 35.2 GiB → 1,151,760 tokens |
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

## The upstream build is faster than this one

Since 2026-11-10 there is a second way to serve this checkpoint: upstream vLLM's
own wheel plus the [`vllm-cmp170hx`](https://github.com/AlexBrandes/vllm) platform
plugin, packaged by [AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next).
It is **8-15% faster on every prompt** here, so the recipe in this repository is
kept for reference and rollback rather than as the recommendation.

Both arms served the same heretic-2 W4A16 checkpoint with the same FP8 PLE, and the
measurement needed a stricter protocol than the one in [Method](#method) — see
[Measuring this model honestly](#measuring-this-model-honestly) below, which is the
most reusable thing in this section. Raw data:
[`results/2026-11-11-sparkdash-upstream-comparison.jsonl`](results/2026-11-11-sparkdash-upstream-comparison.jsonl).

| | this repository | upstream + plugin |
|---|---|---|
| vLLM | `0.1.dev20073+g8e685d198` (a fork; the commit is not in upstream) | `0.33.0rc1.dev659+ga0d7356b9` (official wheel) |
| 170HX support | patch stack + overlays, pinned to `alesha-pro@9b36aae` | `vllm-cmp170hx 0.2.0` platform plugin |
| PLE | `VLLM_PLE_CPU_OFFLOAD=1`, 47 GiB resident in host RAM | `--engram-config {"disk_offload":true}`, `preadv` per step |
| host RAM used | 69-70 GiB | **20 GiB** |

### Decode (tok/s, sparkDash protocol, median of 5, zero stalls)

| prompt | this repository | upstream + nvfp4 | gain |
|---|---|---|---|
| ja prose | 92.8 | **100.3** | +8.1% |
| code | 190.9 | **220.1** | +15.3% |
| en prose | 108.4 | **121.3** | +11.9% |
| ja list | 97.5 | **112.5** | +15.4% |

Both arms run TP2 + expert parallel, `max-num-seqs 12`,
`max-num-batched-tokens 4096`, MTP k=4. The only change on the upstream side is
one speculative-config field:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_lm_head_quantization":"nvfp4"}'
```

`draft_lm_head_quantization` quantizes the drafter's copy of the shared `lm_head`;
the target still verifies with its own head, so acceptance length does not move
(4.855 either way) and output quality cannot change.

### What actually produced the gain

One variable at a time on the upstream image, code prompt, 12 repetitions each:

| step | tok/s | Δ | acceptance | ms/step |
|---|---|---|---|---|
| A  TP2+EP, k=4, nbt 4096, seqs 12, `FULL_DECODE_ONLY` | 209.7 | — | 4.855 | 22.9 |
| B  A + `draft_lm_head_quantization: nvfp4` | 220.6 | **+5.2%** | 4.855 | 21.8 |
| C  B + k=5 + `--block-size 1680` | 230.8 | **+4.6%** | 5.493 | 23.7 |
| D  C + `cudagraph_mode: FULL_AND_PIECEWISE` | 231.9 | +0.5% | 5.493 | 23.6 |
| E  full AlexBrandes config: + PP2, seqs 8, nbt 2048 | 232.2 | +0.1% | 5.493 | 23.5 |

**`nvfp4` and k=5 account for all of it.** Pipeline parallel, `max-num-seqs 8`,
`max-num-batched-tokens 2048` and `FULL_AND_PIECEWISE` contribute nothing
measurable on these cards — which is worth knowing, because PP2 is the headline
difference in the upstream recipe and the expensive one to adopt. (Measured
separately, PP2 is 5% *slower* than TP2+EP here.)

### k=5 is a code-only optimisation

The ablation above used the code prompt, where a longer draft pays. Across all
four prompts it does not:

| prompt | upstream k=4 | upstream k=5 | |
|---|---|---|---|
| ja prose | **100.3** | 95.8 | −4.5% |
| code | 220.1 | **229.7** | +4.4% |
| en prose | **121.3** | 116.1 | −4.3% |
| ja list | **112.5** | 101.8 | −9.5% |

Acceptance is 4.86-5.49 on code and only 2.2-2.9 on prose, so drafting five
tokens ahead wins on code and wastes work everywhere else — the same thing
[Results](#results) already said about larger k, now measured on both builds.
**Keep k=4 for mixed traffic.** k=5 is worth it only for a code-only endpoint,
and then it needs the block size below.

### `k >= 5` needs `--block-size 1680`

Raising k past 4 fails at startup with

```
AssertionError: QSA ring capacity 12 must divide the attention block size 1616
```

`qsa_cache.py` here (`qsa_indexer.py` upstream) sizes the QSA ring as
`round_up(compress_ratio + num_speculative_tokens, compress_ratio)` and asserts
that the attention block size divides it. `compress_ratio` is 4, so k≤4 gives
capacity 8, which the default block size 1616 divides; k=5 gives capacity 12,
which it does not. `--block-size 1680` (1680 / 12 = 140) fixes it. This is why the
upstream recipe carries `--block-size 1680`, and it applies to **this** build too:
`SPEC='{"method":"mtp","num_speculative_tokens":5}' EXTRA_ARGS='--block-size 1680'
./scripts/run.sh` reaches 205.4 tok/s on code without changing images at all.

### `tool_choice` is broken here and fixed there

| | this repository | upstream |
|---|---|---|
| `tool_choice: "required"`, non-streaming, 12 requests | **12 violated** | **0 violated** |
| streaming × thinking, 4 combinations | 1 worked | **4 worked** |
| xgrammar `Failed to advance FSM` + matcher warnings | **215** | **0** in 55 requests |

On this build `tool_choice: "required"` silently degrades to `auto`, and
`finish_reason` is still rewritten to `"tool_calls"` while `tool_calls` stays
empty — a self-contradictory response that breaks any client trusting
`finish_reason`. The cause: `VLLM_ENFORCE_STRICT_TOOL_CALLING` defaults to `True`,
which sets `supports_required_and_named = False` on any parser carrying a
`structural_tag_model` (ours does, `qwen_3_coder`), so required and named tool
choice fall back to the `auto` path; and the grammar that should have enforced it
is built with `reasoning=False` hardcoded, which demands `<tool_call>` as the very
first output and therefore cannot match a thinking model. Upstream's
`ToolStrictLevel` and `constraint_start` resolve both.

### Measuring this model honestly

The numbers above needed three disciplines that [Method](#method) does not use.
Without them this comparison was unrepeatable — the same cell swung 2x — and the
first conclusions drawn from it were wrong.

**Pin the sampling.** MTP acceptance is content-dependent on this model, so
`temperature 0.7` with thinking on and a natural stop length makes every
repetition generate different text at a different length, at a different speed.
The [sparkDash](https://github.com/MiaAI-Lab/sparkDash) protocol at `3e6d4ef` —
the one the upstream recipe publishes its numbers with — pins all three:
`temperature 0`, `top_p 1`, thinking off, and the length fixed with `min_tokens`
plus `ignore_eos`. The tok/s formula is identical to ours. `bench/bench.py
--protocol sparkdash` now does this; the spread drops from 25-80% to 0.4-9.6% and
the output becomes byte-identical between repetitions.

**Gate on an idle engine.** `bench/bench.py --idle-gate` waits for
`num_requests_running` and `num_requests_waiting` to read zero before every
measured request. Anything else reaching the server — the LiteLLM gateway in
`gateway/`, an agent, a dashboard — joins the batch, changes the shape, and
depresses the rate silently. Several readings taken before this existed were
contaminated and had to be thrown away.

**Throw away the first set after a start.** Decode climbs over the first tens of
requests and is erratic until it settles. One measurement set straight after a
start showed the code prompt at 102.9% spread with 10 stalls; re-running it on the
same container, unchanged, gave 0.4% spread and zero stalls. `--warmup 8` helps but
is not sufficient on its own.

A fourth, cheaper note: `docker restart` reaches `/health` in under three minutes
against roughly twelve for a fresh `run`, because the JIT and inductor caches
survive in `models/.vllm-cache`. Use it when sweeping configurations.

### What neither build fixes

Earlier in this investigation single requests stalled badly and unpredictably
(`ITLmax` up to 1,090 ms). With the sampling pinned, 40 consecutive requests
straight after a restart ran 225.7-240.1 tok/s with **zero** stalls, and the
disturbance could not be provoked again: 4-way concurrent bursts, a
20,000-character prefill, thinking-on requests, 1,200-token generations and idle
gaps from 0 to 90 seconds all left it clean. It concentrates in the first tens of
requests after some starts and then goes away.

Ten candidate causes were measured and eliminated: the power cap, swap, major page
faults, PLE host gather, preemption, acceptance variance, prefix caching,
CUDA-graph misses, TP/EP all-to-all, and disk I/O. The decisive observations were
that identical work (same byte-identical output, same step count, same acceptance,
zero preemptions) took 2x the wall time; that the GPU stayed busy at 86-88%
*during* a stall; that the code prompt stalled while reading 0.0 MB from disk; and
that PP2, with no TP, no EP and no collectives, stalled the same. Unexplained.

## Requirements

- **Two 64 GB sm_80 cards.** Only CMP 170HX has been tested.
- **~64 GiB of free host RAM** for the container (61.2 GiB measured). On this 92 GiB host
  that leaves ~31 GiB.
- **~335 GB of disk while preparing**: 179.8 GB base checkpoint + 74 GB heretic-2 +
  51.3 GB derived PLE + 28.8 GB image. The base checkpoint can be deleted once
  `prepare-heretic2.sh` and `build-ple-fp8.sh` have run (~155 GB left).
- Docker with the NVIDIA runtime and a driver new enough for the CUDA 13.0 image.
- If a 170HX runs far below these numbers, check the motherboard `PWRBRK#` power brake
  first — see
  [deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx#troubleshooting-cards-running-4-slow-pwrbrk--edge-pin-b30).
  (Here `HW Power Brake Slowdown` reads `Not Active` on both cards.)

## Quick start

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx
cd qwen38-flash-next-vllm-2x170hx

./scripts/setup.sh              # pinned overlays + vLLM image (28.8 GB)
./scripts/download-model.sh     # base W4A16, 179.8 GB, rate-limited (RATE=95M), resumable
./scripts/build-ple-fp8.sh      # BF16 PLE -> FP8 + manifest
./scripts/download-heretic2.sh  # heretic-2 W4A16, 74 GB, sha256-verified, resumable
./scripts/prepare-heretic2.sh   # GPTQ config, base tokenizer, MTP head graft

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
| `--max-model-len` | 262144 | the model's native maximum; raising it did not slow short requests |
| `--max-num-seqs` | 8 | ~3× aggregate at 4 concurrent; single stream unchanged |
| `--max-num-batched-tokens` | 8192 | works with the KV size below; 4096 returned HTTP 500 with the full-card KV |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | inductor gains nothing here; FULL is downgraded |
| `--kv-cache-memory` | 18897856102 | largest measured size that survives 8192-token prefill batches; vLLM's own suggestion OOMs |
| `--kv-cache-dtype` | `auto` (BF16) | FP8 KV measured: +36% capacity, 6-11% slower decode |
| `VLLM_PLE_EMBEDDING_DTYPE` | `float8_e4m3fn` | selects the FP8 PLE path in the overlay |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | 75 | the 5 s default closed idle connections a LiteLLM gateway was about to reuse ("Server disconnected") |
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

[AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next) authored the
upstream-wheel-plus-plugin build this repository now measures itself against, and the
`vllm-cmp170hx` platform plugin behind it.

- Qwen for the model; [VnimanieAI](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
  for the W4A16 quantization.
- [trohrbaugh](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) for heretic-2;
  [densohax](https://huggingface.co/densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE) for its
  W4A16 quantization.
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
