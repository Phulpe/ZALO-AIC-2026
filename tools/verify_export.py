"""Kiem ban xuat: nguon goc khong doi (source_sha256) va ban chep dung (export_sha256).
Dung: python tools/verify_export.py <thu_muc_ZAIC_goc>"""
import hashlib, json, os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else None
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = json.load(open(os.path.join(here, "EXPORT_MANIFEST.json"), encoding="utf-8"))["files"]
bad_exp = [m["path"] for m in man if sha(os.path.join(here, m["path"])) != m["export_sha256"]]
bad_src = [] if not src else [m["path"] for m in man
           if not os.path.exists(os.path.join(src, m["path"][7:])) or sha(os.path.join(src, m["path"][7:])) != m["source_sha256"]]
print(json.dumps({"files": len(man), "export_mismatch": bad_exp, "source_changed_or_missing": bad_src if src else "not checked"}))
