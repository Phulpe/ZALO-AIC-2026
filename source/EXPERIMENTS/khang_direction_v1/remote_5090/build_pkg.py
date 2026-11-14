"""Dong goi cho may thue RTX 5090 (chay tren may local, Windows). KHONG dua gold len may thue.

Tao <out>/:
  code/            khang_v4 (src, tests, docker, pyproject) + scripts remote_5090/*.sh + fewshot/bank.jsonl
  data_nohint38/   38 cau no-hint TRAIN/calibration (questions.jsonl, calendar.csv, rulebook.md, data/<biz>) - khong gold
  PKG_MANIFEST.json  sha256 tung file
Public (data_public/) chi them khi co co --with-public (can nguoi dung duyet chay public).
--with-adapters: them adapters/<checkpoint-N>/{adapter_model.safetensors,adapter_config.json} cua run
  khang_lora_vast4080_20261112 (chi 2 file inference, khong optimizer/rng/pickle; sha doi chieu FILES_SHA256)
  + ADAPTERS_MANIFEST.json, va code/merge_lora.py.

Dung: python build_pkg.py <out_dir> [--with-public] [--with-adapters]
"""
import hashlib
import json
import os
import shutil
import sys

EXP = "D:/ZAIC/EXPERIMENTS/khang_direction_v1"
PRIV = "D:/ZAIC/PRIVATE_DATA"
RT = f"{PRIV}/relearning_v2_prepared/20261108T070936492119Z"
BANK = f"{PRIV}/khang_direction_v1/fewshot_upload_v1/bank.jsonl"
BANK_SHA = "f74eb15e44e2995fd1131d096f573cd50e59fb2807dbc606f0c2fe7ca1cb630e"
NOHINT = f"{PRIV}/khang_direction_v1/code_upload_v4/nohint_ids.txt"
NOHINT_SHA = "854cecebeedc89a252daa0d6efab81777e848a3195acf77ab50f0e663ed1c368"
PUB = f"{PRIV}/zaic/public_test"  # ban chinh thuc (sha b33ae585, = public v1); khang_v4/data/public_test la ban khac (50e154a2)
PUB_Q_SHA = "b33ae585ed5d4acdacbd6a2873d442e3a5673a8238b6e932827f40c48c158832"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


LORA_RUN = f"{PRIV}/khang_direction_v1/lora_vast4080_20261112"
LORA_CKS = ("checkpoint-20", "checkpoint-40", "checkpoint-85", "checkpoint-127")
FINAL_ADAPTER_SHA = "3bcbd545d03805d46ac7aa61d194d808a5d2a5f97ebdcd9b91c335b87939a08a"


def main(out, with_public, with_adapters=False):
    assert not os.path.exists(out), f"{out} da ton tai (khong ghi de)"
    assert sha(BANK) == BANK_SHA and sha(NOHINT) == NOHINT_SHA
    code = os.path.join(out, "code")
    for d in ("src", "tests", "docker"):
        shutil.copytree(f"{EXP}/khang_v4/{d}", f"{code}/{d}", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(f"{EXP}/khang_v4/pyproject.toml", code)
    os.makedirs(f"{code}/fewshot")
    shutil.copy(BANK, f"{code}/fewshot/bank.jsonl")
    for f in os.listdir(f"{EXP}/remote_5090"):
        if f.endswith(".sh"):
            shutil.copy(f"{EXP}/remote_5090/{f}", f"{code}/{f}")
    ids = set(open(NOHINT).read().split())
    qs = [json.loads(l) for sp in ("learning", "calibration") for l in open(f"{RT}/{sp}/questions.jsonl", encoding="utf-8")]
    sub = sorted([q for q in qs if q["id"] in ids], key=lambda q: q["id"])
    assert len(sub) == len(ids) == 38, (len(sub), len(ids))
    dm = os.path.join(out, "data_nohint38")
    os.makedirs(f"{dm}/data")
    for n in ("calendar.csv", "rulebook.md"):
        shutil.copy(f"{RT}/{n}", dm)
    for b in sorted({q["business_id"] for q in sub}):
        shutil.copytree(f"{RT}/data/{b}", f"{dm}/data/{b}")
    with open(f"{dm}/questions.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in sub))
    if with_public:
        assert sha(f"{PUB}/questions.jsonl") == PUB_Q_SHA
        shutil.copytree(PUB, os.path.join(out, "data_public"))
    ad_man = None
    if with_adapters:
        shutil.copy(f"{EXP}/bridge_v3/merge_lora.py", f"{code}/merge_lora.py")
        ad_files = {}
        for ck in LORA_CKS:
            src = f"{LORA_RUN}/checkpoints/{ck}"
            want = {}
            for line in open(f"{LORA_RUN}/checkpoints/{ck}.FILES_SHA256.txt", encoding="utf-8"):
                h, name = line.split()
                want[name.lstrip("*").replace("./", "")] = h
            os.makedirs(f"{out}/adapters/{ck}")
            for fn in ("adapter_model.safetensors", "adapter_config.json"):
                dst = f"{out}/adapters/{ck}/{fn}"
                shutil.copy(f"{src}/{fn}", dst)
                assert sha(dst) == want[fn], (ck, fn)
                ad_files[f"{ck}/{fn}"] = want[fn]
        assert ad_files["checkpoint-127/adapter_model.safetensors"] == FINAL_ADAPTER_SHA
        ad_man = {"run_id": "khang_lora_vast4080_20261112", "base_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                  "files": ad_files}
        with open(f"{out}/adapters/ADAPTERS_MANIFEST.json", "w", encoding="utf-8") as fh:
            json.dump(ad_man, fh, indent=1)
    files = {}
    for r, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(r, f)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            assert f not in ("gold.jsonl", "answers.jsonl"), rel
            files[rel] = sha(p)
    man = {"files": dict(sorted(files.items())), "bank_sha256": BANK_SHA, "nohint_ids_sha256": NOHINT_SHA,
           "with_public": with_public, "with_adapters": bool(ad_man), "contains_gold": False}
    with open(os.path.join(out, "PKG_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps({"files": len(files), "nohint": len(sub), "with_public": with_public,
                      "manifest_sha256": sha(os.path.join(out, "PKG_MANIFEST.json"))}))


if __name__ == "__main__":
    main(sys.argv[1], "--with-public" in sys.argv[2:], "--with-adapters" in sys.argv[2:])
