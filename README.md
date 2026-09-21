# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[日本語](README_ja.md)

Serving **Qwen3.8-Flash-Next** (W4A16) on two **NVIDIA CMP 170HX** mining cards with
vLLM — GA100, sm_80, VRAM-unlocked to 64 GB, PCIe Gen2 x16, no P2P — on a host with
only **91 GB of RAM**.

**~170–190 tok/s single-stream decode (up to 277 on code) · 1,058,505 KV tokens ·
65K context · tool calling** — about 3.5× the same model as a GGUF on llama.cpp on
one of these cards (49 tok/s).

---

## What is different here

The published Flash-Next W4A16 recipes
([loktar00](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe),
[alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090)) target 4× RTX
3090 and ask for 110–128 GB of host RAM. Two things make it fit on two 170HX and 91 GB:

1. **The PLE n-gram table is converted to FP8 locally.** The W4A16 checkpoints keep the
   51B-parameter per-layer-embedding table in BF16 (~96 GiB), and vLLM's PLE offload
   holds it in host RAM. `scripts/quantize_ple_fp8.py` rewrites it as FP8 E4M3 with one
   global scale (`amax / 448`) — the layout alesha-pro's offload overlay reads — which
   halves it to ~48 GiB. The scale it derives, `0.00019931793212890625`, is
   **bit-identical** to the one in RadixArk's published FP8 PLE, so this is the same
   table without a second 180 GB download.
2. **TP2 + expert parallel on 64 GB cards.** Two 170HX hold the ~77 GiB of GPU-resident
   weights with room for a million-token KV cache. Expert parallel is required: the MoE
   intermediate size 640 split two ways is 320, which the quant's group size of 128 does
   not divide.

MTP speculative decoding (the checkpoint ships its draft head in BF16) is on by default
and is worth about +60%.

## Results

Three prompts (Japanese prose, Python code, a longer list), streaming, `temperature
0.7`, `reasoning_effort: medium`, one sample each. Expect ±15% run to run.

| config | prose | code | longer | median |
|---|---|---|---|---|
| llama.cpp, GGUF Q4_K_M 4.27bpw, 1× 170HX | – | – | – | 49 |
| vLLM W4A16, no MTP | 145.3 | 92.1 | 108.5 | 108.5 |
| vLLM W4A16, MTP k=4 | 173.3 | 276.6 | 149.6 | 173.3 |
| vLLM W4A16, MTP k=4, full KV (**default**) | 188.3 | 253.0 | 143.0 | **188.3** |

Long-form: the 5-section Japanese report prompt in `bench/report-prompt-ja.txt`
produced 6,813 characters / 4,080 tokens in 44 s (102 tok/s, TTFT 4.1 s), finished
cleanly, all sections present, no garbled Japanese.

Raising `--max-num-batched-tokens` from 1024 to 2048 (vLLM warns about it with MTP)
measured 164 vs 173 — noise, so it stays at 1024.

### Where the memory goes

| | per card | total |
|---|---|---|
| weights + non-torch | 38.7 GiB | 77.4 GiB |
| KV cache (BF16) | 22.7 GiB | 45.4 GiB → 1,058,505 tokens |
| host RAM, PLE offload worker | – | 49.2 GiB RSS |

Startup is about 5 minutes: weights ~100 s, FP8 PLE ~15 s, engine init and CUDA graph
capture ~120 s.

The link on these cards negotiates Gen2 x16 and measures **6.6 GB/s** host↔device
(pinned, 256 MB copies), about 83% of the Gen2 x16 ceiling. Some 170HX write-ups
describe the link as fused down to x4; check yours with
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv`
and a copy benchmark, since an x4 card would move a quarter of that.

## Requirements

- **Two 64 GB sm_80 cards.** Tested on CMP 170HX; an A100 80GB pair should work too.
- **~64 GB of free host RAM** for the FP8 PLE worker (49 GiB) plus headroom.
- **~270 GB of disk** (180 GB checkpoint + 51 GB derived PLE + 29 GB image).
- Docker with the NVIDIA runtime, driver new enough for CUDA 13.0.
- If your 170HX throughput is far below these numbers, check for the motherboard
  `PWRBRK#` power brake first — see
  [deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx#troubleshooting-cards-running-4-slow-pwrbrk--edge-pin-b30).

## Quick start

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx
cd qwen38-flash-next-vllm-2x170hx

./scripts/setup.sh            # pinned overlays + vLLM image (~29 GB)
./scripts/download-model.sh   # ~180 GB, rate-limited (RATE=95M), resumable
./scripts/build-ple-fp8.sh    # BF16 PLE -> FP8 + manifest, a few minutes

GPUS=1,2 ./scripts/run.sh     # the two 170HX, by index or UUID (nvidia-smi -L)
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py
```

The endpoint is `http://localhost:18024/v1`, model `flash-next-w4a16`, with the
`qwen3_xml` tool-call parser and `qwen3` reasoning parser enabled.
`gateway/litellm-config.yaml` is an optional LiteLLM front end.

Every path and setting is an environment variable; see `scripts/env.sh` and the top of
`scripts/run.sh`.

## Settings worth knowing

| setting | value | why |
|---|---|---|
| `--tensor-parallel-size 2 --enable-expert-parallel` | required | 640/2 is not divisible by the group size 128 |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | inductor hangs on Ampere for this model; FULL is refused by the QSA backend |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4}` | +60%; `SPEC=` disables |
| `--kv-cache-memory` | 24383208960 | fills a 64 GB card; the fraction-based default leaves ~4.5 GiB idle |
| `--kv-cache-dtype` | `auto` (BF16) | FP8 KV needs a calibration made for this composition |
| `VLLM_PLE_EMBEDDING_DTYPE` | `float8_e4m3fn` | selects the FP8 PLE path in the overlay |
| `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` | required | the PLE offload worker uses `pidfd_getfd` |

## Known limits

- **Send `chat_template_kwargs: {"reasoning_effort": "medium"}` for long outputs.** The
  template defaults to `xhigh`, which can spend tens of thousands of reasoning tokens and
  truncate.
- **Pipeline parallel is not an option.** The PLE offload refuses PP, so on PCIe-only
  cards this is TP with PyNCCL all-reduce (custom all-reduce is disabled without P2P).
- **FP8 KV is untested here.** alesha-pro's calibrated FP8 QSA scales belong to their
  checkpoint; using them on this one is unvalidated.
- **Quality, in one long-form test:** fluent, well-structured Japanese that followed
  every formatting constraint, but it confidently expanded an acronym it did not know
  and filled a comparison table with unsourced numbers. Verify specifics.
- The two cards are fully used; nothing else fits on them alongside this.

## Credits

- Qwen for the model; [VnimanieAI](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
  for the W4A16 quantization.
- [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  for the PLE offload overlays and manifest tooling this mounts (pinned, not vendored).
- [loktar00/qwen38-flash-next-vllm-3090-recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)
  for the measured settings and MTP findings.
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  for the FP8 PLE layout this reproduces.
- vLLM.

## License

Apache-2.0 for the code in this repository. No weights are included; those are under the
Qwen Community License.
