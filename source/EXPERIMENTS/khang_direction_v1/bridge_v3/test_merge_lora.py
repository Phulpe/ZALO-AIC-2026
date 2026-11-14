"""Kiem merge_lora.py tren checkpoint gia lap nho: delta dung B@A*alpha/r, chi sua dung trong so dich, khong dung nhanh visual."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))


def test_merge():
    import torch
    from safetensors.torch import load_file, save_file

    from merge_lora import main

    g = torch.Generator().manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        base, ad, out = (os.path.join(d, x) for x in ("base", "ad", "out"))
        os.makedirs(base); os.makedirs(ad)
        w1 = torch.randn(6, 4, generator=g).to(torch.bfloat16)
        w2 = torch.randn(4, 6, generator=g).to(torch.bfloat16)
        vis = torch.randn(4, 4, generator=g).to(torch.bfloat16)
        save_file({"model.language_model.layers.0.mlp.down_proj.weight": w1}, os.path.join(base, "s1.safetensors"), metadata={"format": "pt"})
        save_file({"model.language_model.layers.1.self_attn.q_proj.weight": w2,
                   "model.visual.blocks.0.mlp.down_proj.weight": vis}, os.path.join(base, "s2.safetensors"), metadata={"format": "pt"})
        json.dump({"weight_map": {"model.language_model.layers.0.mlp.down_proj.weight": "s1.safetensors",
                                  "model.language_model.layers.1.self_attn.q_proj.weight": "s2.safetensors",
                                  "model.visual.blocks.0.mlp.down_proj.weight": "s2.safetensors"}},
                  open(os.path.join(base, "model.safetensors.index.json"), "w"))
        open(os.path.join(base, "config.json"), "w").write("{}")
        a1, b1 = torch.randn(2, 4, generator=g), torch.randn(6, 2, generator=g)
        save_file({"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": a1,
                   "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": b1}, os.path.join(ad, "adapter_model.safetensors"))
        json.dump({"r": 2, "lora_alpha": 4}, open(os.path.join(ad, "adapter_config.json"), "w"))
        main(base, ad, out)
        m1 = load_file(os.path.join(out, "s1.safetensors"))
        m2 = load_file(os.path.join(out, "s2.safetensors"))
        exp = (w1.float() + (b1 @ a1) * 2.0).to(torch.bfloat16)
        assert torch.equal(m1["model.language_model.layers.0.mlp.down_proj.weight"], exp)
        assert torch.equal(m2["model.language_model.layers.1.self_attn.q_proj.weight"], w2)
        assert torch.equal(m2["model.visual.blocks.0.mlp.down_proj.weight"], vis)
        assert os.path.exists(os.path.join(out, "config.json")) and os.path.exists(os.path.join(out, "model.safetensors.index.json"))
        assert torch.equal(load_file(os.path.join(base, "s1.safetensors"))["model.language_model.layers.0.mlp.down_proj.weight"], w1)
    print("OK merge_lora")


if __name__ == "__main__":
    test_merge()
