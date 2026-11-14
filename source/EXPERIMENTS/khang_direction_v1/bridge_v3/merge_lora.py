"""Gop adapter LoRA (PEFT, train tren Qwen3_5ForCausalLM) vao ban sao checkpoint base (Qwen3_5ForConditionalGeneration).

Khong phu thuoc kien truc lop model: doc tung shard safetensors cua base, cong delta = (B @ A) * alpha / r vao trong so
co ten ket thuc bang cung duong dan module (vd "...layers.3.mlp.down_proj.weight", khong dung nhanh visual/mtp),
ghi shard moi vao thu muc dich. Base goc khong bi sua. Kiem: moi module LoRA phai khop dung mot trong so.

Dung: python merge_lora.py <base_dir> <adapter_dir> <out_dir>
"""
import json
import os
import re
import shutil
import sys
import time


def main(base, adapter, out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    A, B = {}, {}
    with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
            assert m, k
            (A if m.group(2) == "A" else B)[m.group(1)] = f.get_tensor(k).float()
    assert A and set(A) == set(B), (len(A), len(B))
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    lang = [k for k in wmap if "visual" not in k and ".mtp." not in k and not k.startswith("mtp.")]
    target = {}
    for mod in A:
        suf = mod.split("layers.", 1)
        assert len(suf) == 2, mod
        key_suf = "layers." + suf[1] + ".weight"
        hits = [k for k in lang if k.endswith("." + key_suf) or k == key_suf]
        assert len(hits) == 1, (mod, hits[:3])
        target[hits[0]] = mod
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(base):
        p = os.path.join(base, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors"):
            shutil.copy2(p, os.path.join(out, fn))
    shards = sorted(set(wmap.values()))
    n_mod = 0
    for sh in shards:
        tensors = {}
        with safe_open(os.path.join(base, sh), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                w = f.get_tensor(k)
                if k in target:
                    mod = target[k]
                    delta = (B[mod] @ A[mod]) * scale
                    assert delta.shape == w.shape, (k, delta.shape, w.shape)
                    w = (w.float() + delta).to(w.dtype)
                    n_mod += 1
                tensors[k] = w.contiguous()
        save_file(tensors, os.path.join(out, sh), metadata=meta or {"format": "pt"})
        del tensors
    assert n_mod == len(target), (n_mod, len(target))
    rep = {"modules": len(target), "r": r, "alpha": alpha, "scale": scale, "shards": len(shards), "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "MERGE_REPORT.json"), "w"), indent=1)
    print("MERGE_DONE:", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
