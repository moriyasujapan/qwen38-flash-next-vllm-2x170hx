#!/usr/bin/env python3
"""Decode, prefill and concurrency benchmark for an OpenAI-compatible server.

    python3 bench/bench.py [--label NAME] [--reps 3] [--prefill 8192,32768]
                           [--conc 1,4,8] [--skip-decode] [--out results.jsonl]
                           [--idle-gate] [--warmup 1]
                           [--protocol default|sparkdash]

decode   : four prompts x --reps, streaming. decode tok/s = (completion_tokens - 1)
           / (t_last_token - t_first_token), so TTFT is excluded. Completion
           tokens include reasoning tokens.
prefill  : synthetic prompts of roughly the given lengths, one output token.
           Each request starts with a random nonce so the prefix cache cannot
           serve it. prefill tok/s = prompt_tokens / TTFT.
conc     : N identical code requests fired together; aggregate tok/s = total
           completion tokens / wall time. Needs --max-num-seqs >= N to mean
           what it says; otherwise requests queue.

--idle-gate waits for /metrics to report no running and no waiting request
before every measured request, and re-checks after a short settle. Use it
whenever anything else can reach the server: co-tenant traffic inflates the
batch, changes the CUDA-graph shape, and silently depresses decode rates.
--warmup N sends N throwaway requests of the real prompt first; one is not
always enough, because decode climbs over the first few requests.

--protocol sparkdash swaps the sampling for the one sparkDash 3e6d4ef uses, the
protocol AlexBrandes publishes his CMP 170HX numbers with: temperature 0, top_p
1, thinking off, and the output length pinned with min_tokens + ignore_eos. The
tok/s formula is unchanged. Pinning all three makes every repetition generate the
same text at the same length, which is what gets the spread down to a few percent
on this model -- MTP acceptance is content-dependent, so under the default
protocol (temperature 0.7, thinking on, natural length) the same cell can swing
2x. Use sparkdash to compare builds or to compare against other 170HX hosts; use
the default protocol for what the deployment actually feels like.

Standard library only.
"""
import argparse
import json
import os
import random
import statistics
import string
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:18024")
MODEL = os.environ.get("MODEL", "flash-next-w4a16")

DECODE_PROMPTS = {
    "ja-prose": "量子もつれを高校生向けに5文で説明して",
    "code": "Pythonで二分探索を書いて。型ヒント付き、コードだけ、説明不要。",
    "en-prose": "Explain how a CPU cache hierarchy works, in two paragraphs.",
    "ja-list": "ローカルLLMを自宅で動かす利点と欠点を、それぞれ5つずつ挙げて",
}
FILLER = ("The quick brown fox jumps over the lazy dog while the committee debates "
          "the budget for next year's infrastructure upgrades in detail. "
          "東京の天気は晴れで、午後から少し雲が広がる見込みです。 ")


PROTOCOL = "default"


def stream(messages, max_tokens, temperature=0.7, pin_length=True):
    """One streaming request. Sampling follows --protocol."""
    if PROTOCOL == "sparkdash":
        # sparkDash 3e6d4ef: greedy, thinking off, output length pinned.
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "top_p": 1, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        if pin_length:
            # Forces exactly max_tokens, so every repetition decodes the same
            # amount of work. Without it the model stops where it likes and the
            # rate moves with the answer.
            body["min_tokens"] = max_tokens
            body["ignore_eos"] = True
    else:
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 0.95, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"reasoning_effort": "medium"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    usage = {}
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                t_first = t_first or time.time()
                t_last = time.time()
    t_end = time.time()
    return {"t0": t0, "ttft": (t_first or t_end) - t0, "t_first": t_first,
            "t_last": t_last, "wall": t_end - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)}


IDLE_GATE = False


def _gauge(text, name):
    for line in text.splitlines():
        if line.startswith(f"vllm:{name}{{"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def wait_idle(timeout=180.0):
    """Block until the server reports no running and no waiting request."""
    if not IDLE_GATE:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
        except Exception:
            time.sleep(0.5)
            continue
        busy = (_gauge(text, "num_requests_running")
                + _gauge(text, "num_requests_waiting"))
        if busy == 0:
            time.sleep(0.4)          # settle, then confirm it stayed idle
            try:
                text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
            except Exception:
                continue
            if _gauge(text, "num_requests_running") == 0:
                return True
        time.sleep(0.5)
    print("  WARNING: server never went idle; this sample is contaminated", flush=True)
    return False


def decode_rate(r):
    n = r["completion_tokens"]
    if not r["t_first"] or n < 2 or r["t_last"] <= r["t_first"]:
        return 0.0
    return (n - 1) / (r["t_last"] - r["t_first"])


def summary(xs):
    med = statistics.median(xs)
    spread = (max(xs) - min(xs)) / med * 100 if med else 0.0
    return {"median": round(med, 1), "min": round(min(xs), 1),
            "max": round(max(xs), 1), "spread_pct": round(spread, 1), "n": len(xs)}


def bench_decode(reps, max_tokens=512):
    per, allr = {}, []
    for name, prompt in DECODE_PROMPTS.items():
        rates = []
        for _ in range(reps):
            wait_idle()
            r = stream([{"role": "user", "content": prompt}], max_tokens)
            rates.append(decode_rate(r))
        per[name] = summary(rates)
        allr += rates
        print(f"  decode {name:9s} {per[name]}", flush=True)
    return {"per_prompt": per, "overall": summary(allr)}


def bench_prefill(lengths, reps=2):
    out = {}
    unit = len(FILLER)
    for target in lengths:
        rates, ptoks, ttfts = [], [], []
        for _ in range(reps):
            wait_idle()
            nonce = "".join(random.choices(string.ascii_letters, k=16))
            # ~3.2 characters per token for this filler; the server reports the
            # real count, which is what the rate uses.
            text = nonce + "\n" + FILLER * max(1, int(target * 3.2 / unit))
            r = stream([{"role": "user",
                         "content": text + "\n\nSummarise the above in one word."}], 1,
                       pin_length=False)
            rates.append(r["prompt_tokens"] / r["ttft"])
            ptoks.append(r["prompt_tokens"])
            ttfts.append(round(r["ttft"], 2))
        out[str(target)] = {"prompt_tokens": ptoks, "ttft_s": ttfts, **summary(rates)}
        print(f"  prefill ~{target:6d} tok (actual {ptoks}) {summary(rates)} tok/s",
              flush=True)
    return out


def bench_conc(levels, conc_max_tokens=256):
    out = {}
    prompt = [{"role": "user", "content": DECODE_PROMPTS["code"]}]
    for n in levels:
        results = [None] * n

        def run(i):
            results[i] = stream(prompt, conc_max_tokens)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        wait_idle()
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        toks = sum(r["completion_tokens"] for r in results)
        out[str(n)] = {"aggregate_tok_s": round(toks / wall, 1),
                       "per_stream_decode_median": round(
                           statistics.median(decode_rate(r) for r in results), 1),
                       "wall_s": round(wall, 1), "tokens": toks}
        print(f"  conc {n}: {out[str(n)]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prefill", default="")
    ap.add_argument("--conc", default="")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--idle-gate", action="store_true",
                    help="wait for an idle server before every measured request")
    ap.add_argument("--warmup", type=int, default=1,
                    help="throwaway requests before measuring (8 is plenty)")
    ap.add_argument("--protocol", choices=("default", "sparkdash"), default="default",
                    help="sampling protocol; sparkdash pins text and length")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="decode output length (default 512, sparkdash 400)")
    a = ap.parse_args()

    global IDLE_GATE, PROTOCOL
    IDLE_GATE = a.idle_gate
    PROTOCOL = a.protocol
    max_tokens = a.max_tokens or (400 if PROTOCOL == "sparkdash" else 512)

    # Throwaway requests so the first measured one is not paying warmup. Decode
    # climbs over the first few requests, so one is not always enough.
    warmup_tokens = 32 if PROTOCOL == "sparkdash" else 256
    for _ in range(max(0, a.warmup)):
        stream([{"role": "user", "content": DECODE_PROMPTS["code"]}], warmup_tokens)
    res = {"label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "protocol": PROTOCOL, "max_tokens": max_tokens,
           "idle_gate": IDLE_GATE, "warmup": a.warmup}
    if not a.skip_decode:
        res["decode"] = bench_decode(a.reps, max_tokens)
    if a.prefill:
        res["prefill"] = bench_prefill([int(x) for x in a.prefill.split(",")])
    if a.conc:
        res["conc"] = bench_conc([int(x) for x in a.conc.split(",")], max_tokens)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
