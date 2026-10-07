"""Make the heretic-2 AutoRound W4A16 checkpoint load in the pinned vLLM image, with an MTP head.

Usage: prepare_heretic2.py <heretic2 dir> <base W4A16 dir>. Idempotent: originals are kept beside the
files it rewrites (*.autoround.orig, *.heretic.orig, *.index.json.orig, orig-tokenizer/) and are what
every rerun starts from.

The checkpoint is AutoRound W4A16 (`quant_method: auto-round`, `auto_round:auto_gptq` packing): only the
routed experts carry qweight/qzeros/scales, every other weight is BF16. The image registers no `auto-round`
method, but its AutoGPTQ config reads the same packing:
  - Linear layers: with `modules_in_block_to_quantize` empty, is_layer_gptq_quantized() is False for every
    Linear, so they all load unquantized, which is what this checkpoint stores.
  - Routed experts: quantized unless a `dynamic` negative pattern matches. The grafted MTP head's experts are
    BF16, so `-:.*mtp.*` keeps them unquantized.
config.json is otherwise the base W4A16 one this image already serves (same architecture; the heretic
config only adds keys the image does not know, e.g. vision model_type qwen4_exp_vision).

The tokenizer and chat-template files are replaced with the base checkpoint's, which the image's
`qwen3_xml` tool parser and `qwen3` reasoning parser were set up against, and generation_config.json gets
the base model's stop tokens (248046 and 248044) and sampling defaults; the heretic file stops on 248044
only.

The checkpoint ships no MTP tensors, so the base model's BF16 MTP head is copied from the base checkpoint
into model-mtp-graft.safetensors. It was trained against the unmodified base, so draft acceptance may be a
little lower; verification keeps the target model's tokens either way.
"""
import filecmp
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

DST, BASE = sys.argv[1], sys.argv[2]
GRAFT = "model-mtp-graft.safetensors"
TOKENIZER_FILES = [
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "video_preprocessor_config.json",
]
GENERATION_CONFIG = {
    "bos_token_id": 248044,
    "do_sample": True,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}


def keep_original(path, suffix):
    if os.path.exists(path) and not os.path.exists(path + suffix):
        shutil.copy2(path, path + suffix)
    return path + suffix


def rewrite_config():
    cfg_path = os.path.join(DST, "config.json")
    heretic = json.load(open(keep_original(cfg_path, ".autoround.orig")))
    config = json.load(open(os.path.join(BASE, "config.json")))
    src_q = heretic["quantization_config"]
    assert src_q["quant_method"] == "auto-round" and src_q["packing_format"] == "auto_round:auto_gptq"
    assert src_q["bits"] == 4 and src_q["group_size"] == 128 and src_q["sym"] is True
    config["quantization_config"] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "checkpoint_format": "gptq",
        "dynamic": {"-:.*mtp.*": {}},
        "autoround_version": src_q.get("autoround_version", ""),
    }
    json.dump(config, open(cfg_path, "w"), indent=2)
    # A stray quantization_config.json would be picked up instead of config.json's.
    qc = os.path.join(DST, "quantization_config.json")
    if os.path.exists(qc):
        os.replace(qc, qc + ".autoround.orig")
    print("config.json: gptq w4g128 sym, experts only, MTP excluded")


def replace_tokenizer():
    orig = os.path.join(DST, "orig-tokenizer")
    os.makedirs(orig, exist_ok=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(BASE, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(DST, name)
        # A file already equal to the base one was put there by an earlier run (or never differed).
        if os.path.exists(dst) and not filecmp.cmp(src, dst, shallow=False) \
                and not os.path.exists(os.path.join(orig, name)):
            shutil.copy2(dst, os.path.join(orig, name))
        shutil.copy2(src, dst)
    gen_path = os.path.join(DST, "generation_config.json")
    keep_original(gen_path, ".heretic.orig")
    json.dump(GENERATION_CONFIG, open(gen_path, "w"), indent=2)
    print("tokenizer, chat template and generation_config.json: base model's")


def graft_mtp():
    base_map = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))["weight_map"]
    mtp_keys = sorted(k for k in base_map if k.startswith("mtp."))
    by_file = {}
    for k in mtp_keys:
        by_file.setdefault(base_map[k], []).append(k)
    tensors = {}
    for fname, keys in by_file.items():
        with safe_open(os.path.join(BASE, fname), framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    save_file(tensors, os.path.join(DST, GRAFT), metadata={"format": "pt", "source": "base W4A16 mtp.*"})
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{GRAFT}: {len(tensors)} tensors, {size / 2**30:.2f} GiB")

    index_path = os.path.join(DST, "model.safetensors.index.json")
    index = json.load(open(keep_original(index_path, ".orig")))
    assert not any(k.startswith("mtp.") for k in index["weight_map"])
    for k in mtp_keys:
        index["weight_map"][k] = GRAFT
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + size
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"index: +{len(mtp_keys)} mtp entries")


if __name__ == "__main__":
    rewrite_config()
    replace_tokenizer()
    graft_mtp()
