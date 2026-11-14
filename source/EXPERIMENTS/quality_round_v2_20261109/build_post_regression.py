"""Prepare bounded read-only GPU evaluation; never starts it."""
from pathlib import Path
import json, hashlib

here=Path(__file__).parent
folder=here/'POST_REGRESSION_GPU'; folder.mkdir(exist_ok=True)
prefix=(here/'FINISH_GPU/finish.py').read_text(encoding='utf-8').split('CONFIG=')[0]
script=(here/'post_learning_regression.py').read_text(encoding='utf-8')
config=json.loads((here/'FINISH_GPU_REQUEST_LOCAL.json').read_text(encoding='utf-8'))
cache=Path('D:/ZAIC/PRIVATE_DATA/KAGGLE_RUNS/trusted_train_cache/TRUSTED_TRAIN_CACHE')
cache_manifest=hashlib.sha256((cache/'ARTIFACT_MANIFEST.json').read_bytes()).hexdigest()
tail='''
sys.path.insert(0,str(root))
from relearning import verify_export,sha
post=root/'post_learning_regression.py'; post.write_text(POST_SCRIPT,encoding='utf-8')
assert sha(post)==POST_SCRIPT_SHA
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
collections=[p.parent for p in Path('/kaggle/input').rglob('QUALITY_GATE_RESULT.json') if p.parent.name==CONFIG['verified_collection_run_id']]
receipts=list(Path('/kaggle/input').rglob('CANDIDATE_RECEIPT.json'))
caches=list(Path('/kaggle/input').rglob('CACHE_PROVENANCE.json'))
assert len(bases)==len(runtimes)==len(collections)==len(receipts)==len(caches)==1
receipt=json.loads(receipts[0].read_text());original=Path(receipt['run_path'])
candidate=receipts[0].parent/'model_runs'/original.name/'checkpoint'
assert candidate.exists() and sha(candidate/'adapter_model.safetensors')==receipt['adapter_sha256']
assert verify_export(candidate.parent)==receipt['manifest_sha256']
CONFIG.update(mode='evaluation',base=str(bases[0].parent/'base'),runtime_root=str(runtimes[0].parent),verified_collection=str(collections[0]),candidate_adapter=str(candidate),candidate_sha256=receipt['adapter_sha256'],regression_output='/kaggle/working/regression_output',trusted_cache_root=str(caches[0].parent),trusted_cache_manifest_sha256=CACHE_MANIFEST_SHA)
request=root/'POST_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
for name,digest in CONFIG['base_hashes'].items():assert sha(Path(CONFIG['base'])/name)==digest
with Path('/kaggle/working/POST_PRIVATE.log').open('w') as log:
 result=subprocess.run([sys.executable,str(post),str(request)],cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=POST_TIMEOUT)
assert result.returncode==0,'Read-only regression failed; retain private log'
print(Path(CONFIG['regression_output']).joinpath('REGRESSION_SUMMARY.json').read_text())
'''
body=prefix+'\nCONFIG='+repr(config)+'\nPOST_SCRIPT='+repr(script)+'\nPOST_SCRIPT_SHA='+repr(hashlib.sha256(script.encode()).hexdigest())+'\nCACHE_MANIFEST_SHA='+repr(cache_manifest)+'\nPOST_TIMEOUT=900\n'+tail
(folder/'post.py').write_text(body,encoding='utf-8')
metadata=json.loads((here/'FINISH_GPU/kernel-metadata.json').read_text())
metadata.update(id='lnghongphc/zaic-quality-post-regression-20261109',title='ZAIC Quality Post Regression 20261109',code_file='post.py',kernel_sources=['lnghongphc/zaic-quality-round-v2-20261109/3','lnghongphc/zaic-quality-train-eval-public-20261109/1','lnghongphc/zaic-trusted-train-cache-20261109/1'])
metadata['dataset_sources']=[s for s in metadata['dataset_sources'] if 'base-' in s or 'runtime-' in s or 'bnb-' in s]
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps({'state':'prepared_not_started','post_sha256':hashlib.sha256(script.encode()).hexdigest(),'cache_manifest_sha256':cache_manifest,'timeout_seconds':900}))
