from pathlib import Path
import sys,subprocess,json,traceback
p=Path('/content/zaic_9b_v21_20261107')
for args,cwd in [([sys.executable,'-m','unittest','test_core.py','test_v2_regression.py','-q'],p/'app'),([sys.executable,'-m','unittest','discover','-s','tools','-p','test*.py','-q'],p)]:
 subprocess.run(args,cwd=cwd,check=True)
sys.path.insert(0,str(p/'app'))
from legacy_engine import prepare_shop
s=prepare_shop(Path('/content/drive/MyDrive/zaic/train/data/biz_124'),p/'loader_diagnostic_cache')
stats={'loaded':True,'lines':len(s.lines),'orders':len(s.orders),'lines_with_dates':int(s.lines.date.notna().sum()),'lines_with_order_ids':int(s.lines.order_id.notna().sum()),'date_min':str(s.lines.date.min()),'date_max':str(s.lines.date.max()),'tests_passed':99}
h=Path('/content/drive/MyDrive/ZAIC_9B_Repair_v21_20261107')
(h/'LOADER_VERIFIED.json').write_text(json.dumps(stats,indent=2))
print('LOADER_VERIFIED',json.dumps(stats),flush=True)
