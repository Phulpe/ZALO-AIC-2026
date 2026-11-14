"""Pinned adapter loading and TRAIN-only field diagnostics; no optimizer."""
from pathlib import Path
import json,collections
from relearning import sha,verify_export,atomic,seal

def verify_loaded(model,adapter,adapter_name='default'):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected=load_file(str(Path(adapter)/'adapter_model.safetensors'),device='cpu')
    actual=get_peft_model_state_dict(model,adapter_name=adapter_name)
    if set(actual)!=set(expected):raise ValueError('Adapter tensor keys differ after loading')
    for key,value in expected.items():
        loaded=actual[key].detach().cpu()
        if loaded.shape!=value.shape or not torch.equal(loaded.float(),value.float()):raise ValueError('Loaded adapter tensor differs from source')
    return {'adapter_sha256':sha(Path(adapter)/'adapter_model.safetensors'),'all_tensor_values_match':True,'tensors':len(expected)}

def teacher_stats(model,tok,rows):
    import torch
    from time_objective import training_classes,weighted_token_loss
    dataset,collate,_=training_classes(1.);ds=dataset(rows,tok);totals=collections.defaultdict(collections.Counter)
    model.eval()
    with torch.inference_mode():
        for row,item in zip(rows,ds.items):
            data={k:v.to('cuda:0') for k,v in collate([item]).items()};labels=data.pop('labels');weights=data.pop('loss_weights');focus=data.pop('time_focus');n=int(labels.ne(-100).sum().item())
            outputs=model(**data,logits_to_keep=n+1,use_cache=False)
            _,stats=weighted_token_loss(outputs.logits[:,:-1,:],labels[:,-n:],weights[:,-n:],focus[:,-n:])
            totals[row['kind']].update(stats)
    result={}
    for kind,s in totals.items():result[kind]={**dict(s),'mean_nll':s['nll_sum']/s['tokens'],'time_mean_nll':s['time_nll_sum']/s['time_tokens'],'time_token_accuracy':s['time_correct_tokens']/s['time_tokens']}
    return {'scope':'teacher-forcing on previously verified TRAIN targets; not blind accuracy','weights_updated':False,'rows':len(rows),'groups':result}

class TrustedShops:
    def __init__(self,root,manifest_sha,runtime_sha):
        import importlib.metadata
        self.root=Path(root);assert verify_export(self.root)==manifest_sha
        self.meta=json.loads((self.root/'CACHE_PROVENANCE.json').read_text());assert self.meta['runtime_manifest_sha256']==runtime_sha
        assert self.meta['parser_sha256']==sha(Path(__file__).parent/'baseline/app/legacy_engine.py')
        assert self.meta['pandas_version']==importlib.metadata.version('pandas');self.current=None;self.shop=None
    def __getitem__(self,biz):
        import pickle
        if self.current!=biz:
            meta=self.meta['files'][biz];p=(self.root/meta['path']).resolve();assert p.is_relative_to(self.root.resolve()) and sha(p)==meta['sha256']
            with p.open('rb') as stream:self.shop=pickle.load(stream)
            self.current=biz
        return self.shop

def evaluate_rows(model,tok,rows,pool,gold,shops,out,stop_requested=lambda:False):
    from baseline_bridge import BaselineBridge,matches
    from prediction_errors import BlindPredictionError
    out=Path(out);out.mkdir(exist_ok=False);qs={q['id']:q for q in pool};records={};verdicts={};bridge=BaselineBridge(model,tok,shops,max_tokens=240,max_seconds=25)
    for row in sorted(rows,key=lambda r:(r['business_id'],r['id'])):
        if stop_requested():raise TimeoutError('Budget exhausted before evaluation question')
        q=qs[row['id']];rec={'state':'attempting'};records[row['id']]=rec;atomic(out/'PRIVATE_PREDICTIONS.json',records)
        try:
            plan=bridge.predict(q,1);rec.update(state='predicted',plan=plan);atomic(out/'PRIVATE_PREDICTIONS.json',records)
            verdict=bridge.semantic(q,plan) and matches(bridge.execute(q,plan),gold(q['id']))
        except Exception as exc:
            verdict=False;rec.update(state='error',exception_type=type(exc).__name__)
            if isinstance(exc,BlindPredictionError):rec['prediction_text']=exc.prediction_text
        verdicts[q['id']]=bool(verdict);atomic(out/'PRIVATE_PREDICTIONS.json',records)
    corr=[r['id'] for r in rows if r['kind']=='correction'];replay=[r['id'] for r in rows if r['kind']=='replay']
    report={'mode':'evaluation','scope':'learning regression; not holdout','expected':len(rows),'attempted':len(verdicts),'real_output_ids':sum('plan' in r or 'prediction_text' in r for r in records.values()),'correction_before_correct':0,'correction_after_correct':sum(verdicts[i] for i in corr),'correction_total':len(corr),'replay_before_correct':len(replay),'replay_after_correct':sum(verdicts[i] for i in replay),'correct_to_wrong':sum(not verdicts[i] for i in replay),'weights_updated':False}
    atomic(out/'REGRESSION_SUMMARY.json',report);seal(out);return report
