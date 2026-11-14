from pathlib import Path
import hashlib, json

def input_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

root=Path('/kaggle/input')
adapters=list(root.rglob('final/adapter_model.safetensors'))
assert len(adapters)==1, 'Expected one trained input'
learned=adapters[0].parent.parent
expected_adapter='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
assert adapters[0].stat().st_size==86623432
assert input_sha(adapters[0])==expected_adapter
expected_checkpoint={'adapter_config.json':'81a8f3b6bb0fea3694645faecbd639b546d852178d9b5392bf5fb9cc08380abc','adapter_model.safetensors':expected_adapter,'chat_template.jinja':'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715','optimizer.pt':'2cd52041406af338fb8c0331e9facda39d1efb093f068331f326611af0e6e03b','README.md':'40cf28822b7f9e697cd8b498aa396a84c6146e6d7f4a894937286413e4232ba5','rng_state.pth':'3f4a6e311653bed0ebcbed5bc7b638bd0e12c196803494f9098d770fb2fd7bce','scaler.pt':'ef4a091e18e7e3127686face18428e35b3700c95f401ba63fb35b5aa6ba828ca','scheduler.pt':'8f8c15637bdf0a16f4962ec44b7231f03100e3eae1261ae580198c04915f691c','tokenizer.json':'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f','tokenizer_config.json':'6fc30221b0ec773811dfa2a04ebf0eec1a7e36a3bb1698888e28b38eabc1126c','trainer_state.json':'2b23b0e119734a1eec535a70ddee2b2b5a047a45a2ebdbf91b5a8066f235549f','training_args.bin':'ec304bce42943a583fdf95233a19e69e9c5e0fef14e0d40c05930fabc800447d'}
for name,h in expected_checkpoint.items():
    p=learned/'checkpoints/checkpoint-48'/name
    assert p.is_file() and input_sha(p)==h, 'Checkpoint hash mismatch'
assert input_sha(learned/'model_lock.json')=='6c7616ebe581cbf90a3b0d78f95631f4c907c1e65953f7c796375271447b2a21'
locks=json.loads((learned/'model_lock.json').read_text())
assert locks['model']=='Qwen/Qwen3.5-9B' and locks['revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
sft_manifests=list(root.rglob('SFT_UPLOAD_MANIFEST.json'))
assert len(sft_manifests)==1
sft_root=sft_manifests[0].parent/'data'
expected_sft={'manifest.json':'27eb28bf933fa37f7cb6802d4c9af5779a86cc5227a6ed691c7da8eb21452603','sft_train.jsonl':'5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170','split.json':'4f12303bf02a82ee10d86994c3d6c8ea293517523dcfc8ef9065f10e07707bd5'}
for name,h in expected_sft.items():
    assert input_sha(sft_root/name)==h, 'SFT input hash mismatch'
receipt={'trained_final_adapter':'sha256 matched','checkpoint48_files_sha256_matched':len(expected_checkpoint),'model_revision':'matched','sft_files_sha256_matched':len(expected_sft),'base':'upload pending; not checked','training_started':False}
Path('/kaggle/working/INPUT_INTEGRITY_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
