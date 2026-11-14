"""Windows working-copy preflight; no edits to frozen_v21 or security settings."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from kaggle_runner import verify_frozen, environment_report

project = Path(__file__).resolve().parents[1]
frozen = project / 'frozen_v21'
assert verify_frozen(frozen) == 27
working = project / 'experiments/windows_portable_v1'
working.mkdir(parents=True, exist_ok=True)
for folder in ['app','tools']:
    shutil.copytree(frozen/folder, working/folder, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
path = working / 'tools/infer_batched.py'
text = path.read_text(encoding='utf-8')
needle = 'str(f.relative_to(root))'
assert text.count(needle) == 1
path.write_text(text.replace(needle, 'f.relative_to(root).as_posix()'), encoding='utf-8')
test = working / 'tools/test_batch_io.py'
text = test.read_text(encoding='utf-8')
needle = "alias.symlink_to(root,target_is_directory=True)"
assert text.count(needle) == 1
replacement = "(__import__('_winapi').CreateJunction(str(root),str(alias)) if __import__('os').name=='nt' else alias.symlink_to(root,target_is_directory=True))"
test.write_text(text.replace(needle, replacement), encoding='utf-8')
program = '''import sys,io,unittest,json
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p/'app'),str(p/'tools')]
s=unittest.TestLoader().loadTestsFromNames(['test_core','test_v2_regression','test_audit_output','test_batch_io','test_policy'])
r=unittest.TextTestRunner(stream=io.StringIO(),verbosity=0).run(s)
print(json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'passed':r.wasSuccessful(),'error_types':[e.splitlines()[-1] for _,e in r.errors]}))
'''
result = subprocess.run([sys.executable,'-X','utf8','-c',program,str(working)],
                        env={**os.environ,'PYTHONUTF8':'1'},capture_output=True,text=True,encoding='utf-8')
summary = json.loads(result.stdout.strip().splitlines()[-1])
report = {'frozen_source_hashes_verified':verify_frozen(frozen),
    'scope':'Working Windows copy. Frozen Linux code remains unchanged.',
    'changes':['POSIX separators in fingerprint keys',
               'Use existing Windows directory-junction API in path-alias test; no privileged symlink permission'],
    'synthetic_tests':summary,'environment':environment_report()}
(project/'reports/WINDOWS_PORTABLE_PREFLIGHT_20261107.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if not summary['passed'] or summary['tests'] != 99 or summary['skipped']:
    raise SystemExit(1)
