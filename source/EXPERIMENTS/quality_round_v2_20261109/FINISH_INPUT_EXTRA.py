
sys.path[:0]=[str(root),str(root/'baseline/app'),str(root/'baseline/tools')]
from relearning import sha,verify_export,verify_parent,atomic
from quality_gate import select
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'));adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'));runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'));collections=list(Path('/kaggle/input').rglob('QUALITY_GATE_RESULT.json'));bundles=list(Path('/kaggle/input').rglob('TRAIN_PROVENANCE.json'))
assert len(bases)==len(adapters)==len(runtimes)==len(collections)==len(bundles)==1
CONFIG.update(base=str(bases[0].parent/'base'),adapter=str(adapters[0].parent),runtime_root=str(runtimes[0].parent),verified_collection=str(collections[0].parent),public_root=str(bundles[0].parent/'PUBLIC'))
assert Path(CONFIG['verified_collection']).name==CONFIG['verified_collection_run_id']
assert verify_export(Path(CONFIG['verified_collection']))==CONFIG['collection_manifest_sha256']
verify_parent(CONFIG['adapter'],CONFIG['parent_sha256'])
for name,h in CONFIG['base_hashes'].items():assert sha(Path(CONFIG['base'])/name)==h
assert sha(Path(CONFIG['runtime_root'])/'calibration/questions.jsonl')==CONFIG['calibration_questions_sha256']
assert sha(Path(CONFIG['runtime_root'])/'calibration/gold.jsonl')==CONFIG['calibration_gold_sha256']
assert verify_export(bundles[0].parent)==CONFIG['bundle_manifest_sha256']
source=Path(CONFIG['verified_collection']);bank=json.loads((source/'verified_bank.json').read_text());evidence=json.loads((source/'PRIVATE_TARGET_EVIDENCE.json').read_text());rows=[row for row in bank['correction']+bank['replay'] if row['id'] in evidence]
selected,selection=select(rows,evidence,CONFIG['supervision_gate'])
from transformers import AutoTokenizer
from train import CompletionDataset
tok=AutoTokenizer.from_pretrained(CONFIG['base'],local_files_only=True,trust_remote_code=False)
ds=CompletionDataset(selected,tok,2048)
assert not ds.skipped and len(ds)==len(selected)
for item in ds:
 labels=item['labels'];ids=item['input_ids'];assert len(ids)==len(labels) and labels[0]==-100 and ids[-1]==tok.eos_token_id
 first=next(i for i,v in enumerate(labels) if v!=-100)
 assert first>0 and all(v==-100 for v in labels[:first]) and labels[first:]==ids[first:]
 assert sum(v!=-100 for v in labels[1:])==len(labels)-first

atomic(Path('/kaggle/working/FINISH_INPUT_CHECK.json'),{'state':'passed','selection':selection,'token_rows':len(ds),'overlength_rows':ds.skipped,'base_files_verified':len(CONFIG['base_hashes']),'collection_manifest_sha256':CONFIG['collection_manifest_sha256']})
print('FINISH_INPUT_CHECK_PASSED')
