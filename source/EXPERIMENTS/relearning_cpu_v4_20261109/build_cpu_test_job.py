"""Deploy code/synthetic tests only to Kaggle CPU. No competition input mount."""
from pathlib import Path
import zipfile,json,hashlib,base64,sys
revision=sys.argv[1] if len(sys.argv)>1 else '1';expected={'1':264,'2':267,'3':268,'4':274,'5':282}[revision]
exp=Path(__file__).parent;src=exp/'src';deploy=exp/('KAGGLE_CPU_TESTS' if revision=='1' else 'KAGGLE_CPU_TESTS_R'+revision);deploy.mkdir(exist_ok=False)
parent=json.loads((exp/'SOURCE_COPY_RECEIPT.json').read_text(encoding='utf-8'));baseline={name:digest for name,digest in parent['files'].items() if name.startswith('baseline/')}
for name,digest in baseline.items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest
archive=exp/('CODE_V4_CPU_TESTED.zip' if revision=='1' else 'CODE_V4_R'+revision+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(src.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(src).as_posix())
payload=archive.read_bytes();digest=hashlib.sha256(payload).hexdigest()
code='''from pathlib import Path
import json,sys,os,hashlib,base64,zipfile,io,stat,unittest,importlib.metadata,platform
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
payload=base64.b64decode(PAYLOAD);assert hashlib.sha256(payload).hexdigest()==SOURCE_HASH
src=Path('/kaggle/working/source');src.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 for p in z.infolist():assert (src/p.filename).resolve().is_relative_to(src.resolve()) and not stat.S_ISLNK(p.external_attr>>16)
 z.extractall(src)
sys.path[:0]=[str(src),str(src/'baseline/app'),str(src/'baseline/tools')]
for name,digest in BASELINE.items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest
results={};report=Path('/kaggle/working/TEST_DETAILS.txt')
with report.open('w',encoding='utf-8') as stream:
 for name,path in [('app',src/'baseline/app'),('tools',src/'baseline/tools'),('new',src)]:
  suite=unittest.TestLoader().discover(str(path),pattern='test*.py');r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);results[name]={'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'passed':r.wasSuccessful()}
summary={'source_archive_sha256':SOURCE_HASH,'baseline_python_bytes_unchanged':True,'results':results,'environment':{'python':sys.version,'platform':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','pandas','numpy','openpyxl']}},'gpu_executed':False,'optimizer_executed':False,'competition_data_mounted':False}
Path('/kaggle/working/TEST_RESULTS.json').write_text(json.dumps(summary,indent=2));print('CPU_TEST_SUMMARY',json.dumps(summary))
assert sum(r['tests'] for r in results.values())==EXPECTED_TESTS and all(r['passed'] and not r['skipped'] for r in results.values()),'CPU tests failed; preserve test report'
'''
code='PAYLOAD='+repr(base64.b64encode(payload).decode())+'\nSOURCE_HASH='+repr(digest)+'\nBASELINE='+repr(baseline)+'\nEXPECTED_TESTS='+str(expected)+'\n'+code
(deploy/'cpu_tests.py').write_text(code,encoding='utf-8');compile(code,'cpu_tests.py','exec')
metadata={'id':'lnghongphc/zaic-relearning-v4-cpu-tests-20261110','title':'ZAIC Relearning v4 CPU Tests 20261110','code_file':'cpu_tests.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]};(deploy/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
(exp/('CPU_DEPLOYMENT_RECEIPT.json' if revision=='1' else 'CPU_DEPLOYMENT_RECEIPT_R'+revision+'.json')).write_text(json.dumps({'source_archive_sha256':digest,'wrapper_sha256':hashlib.sha256(code.encode()).hexdigest(),'baseline_files_verified':len(baseline),'expected_tests':expected,'gpu':False,'raw_data_included':False},indent=2),encoding='utf-8');print('CPU_CODE_ONLY_READY',digest)
