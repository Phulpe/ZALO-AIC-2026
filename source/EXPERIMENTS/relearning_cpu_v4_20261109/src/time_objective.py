"""Completion-only temporal objective and aggregate diagnostics; no gold access."""
import json,math

VERSION='temporal_payload_completion_v1'

def time_span(response):
    """Locate the top-level time value, rejecting ambiguous duplicate JSON keys."""
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    decoder=json.JSONDecoder(object_pairs_hook=unique)
    obj=decoder.decode(response)
    if not isinstance(obj,dict) or not isinstance(obj.get('time'),dict):raise ValueError('Time object required')
    pos=0
    def ws(i):
        while i<len(response) and response[i].isspace():i+=1
        return i
    pos=ws(pos)
    if response[pos]!='{':raise ValueError('Plan object required')
    pos+=1;found=None
    while True:
        pos=ws(pos)
        if response[pos]=='}':break
        key,pos=decoder.raw_decode(response,pos);pos=ws(pos)
        if response[pos]!=':':raise ValueError('Missing colon')
        start=ws(pos+1);_,end=decoder.raw_decode(response,start)
        if key=='time':found=(start,end)
        pos=ws(end)
        if response[pos]=='}':break
        if response[pos]!=',':raise ValueError('Missing comma')
        pos+=1
    if found is None:raise ValueError('Time span missing')
    return found

def token_focus(offsets,span,response_length):
    start,end=span
    if not 0<=start<end<=response_length:raise ValueError('Invalid time span')
    result=[]
    for a,b in offsets:
        if not 0<=a<=b:raise ValueError('Invalid token offsets')
        result.append(a<b and a<end and b>start and a<response_length)
    if not any(result):raise ValueError('No time tokens')
    return result

def payload_spans(response):
    """Focus semantic values, excluding fixed time keys and object punctuation."""
    start,end=time_span(response);decoder=json.JSONDecoder();pos=start+1;spans=[]
    def ws(i):
        while response[i].isspace():i+=1
        return i
    while True:
        pos=ws(pos)
        if response[pos]=='}':break
        key,pos=decoder.raw_decode(response,pos);pos=ws(pos)
        if response[pos]!=':':raise ValueError('Missing time colon')
        a=ws(pos+1);value,b=decoder.raw_decode(response,a)
        if key not in {'kind','value','date','start','end','n'} or isinstance(value,(list,dict)) or value is None:raise ValueError('Unexpected time payload')
        # Quotes are excluded, but tokenizer boundary overlap is intentional.
        spans.append((a+1,b-1) if isinstance(value,str) else (a,b));pos=ws(b)
        if response[pos]=='}':break
        if response[pos]!=',':raise ValueError('Missing time comma')
        pos+=1
    if not spans or any(a>=b for a,b in spans):raise ValueError('Empty time payload')
    return spans

def payload_focus(offsets,response):
    result=[False]*len(offsets)
    for span in payload_spans(response):result=[a or b for a,b in zip(result,token_focus(offsets,span,len(response)))]
    return result

def validate_weight(weight):
    if isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or not 1<=weight<=8:raise ValueError('Time weight must be finite in [1,8]')
    return float(weight)

def weighted_token_loss(logits,targets,weights,focus):
    import torch
    import torch.nn.functional as F
    if logits.ndim!=3 or targets.shape!=logits.shape[:2] or weights.shape!=targets.shape or focus.shape!=targets.shape:raise ValueError('Token alignment mismatch')
    if targets.lt(0).any() or not torch.isfinite(weights).all() or weights.le(0).any() or not focus.any():raise ValueError('Completion mask/weights invalid')
    ce=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='none').reshape_as(targets)
    loss=(ce*weights).sum()/weights.sum()
    correct=logits.argmax(-1).eq(targets)
    stats={'tokens':targets.numel(),'time_tokens':int(focus.sum().item()),'nll_sum':float(ce.detach().sum().item()),'time_nll_sum':float(ce.detach()[focus].sum().item()),'time_correct_tokens':int(correct[focus].sum().item()),'all_correct_tokens':int(correct.sum().item())}
    return loss,stats

def training_classes(weight):
    """Lazy torch imports permit pure CPU audits without installing a GPU stack."""
    weight=validate_weight(weight)
    import torch
    from train import CompletionTrainer
    from modeling import chat
    from planner import SYSTEM,validate_plan
    class TemporalDataset(torch.utils.data.Dataset):
        def __init__(self,rows,tok,maxlen=2048):
            if not tok.is_fast:raise ValueError('Fast tokenizer offsets required')
            self.items=[];self.skipped=0
            for row in rows:
                validate_plan(json.loads(row['response']))
                prefix=tok(chat(tok,SYSTEM,row['prompt']),add_special_tokens=False)['input_ids']
                encoded=tok(row['response']+tok.eos_token,add_special_tokens=False,return_offsets_mapping=True)
                target=encoded['input_ids'];focus=payload_focus(encoded['offset_mapping'],row['response'])
                if len(prefix)+len(target)>maxlen:raise ValueError('Overlength target; no silent dropping')
                if not prefix or not target or target[-1]!=tok.eos_token_id:raise ValueError('Prefix/EOS contract failed')
                self.items.append({'input_ids':prefix+target,'labels':[-100]*len(prefix)+target,'loss_weights':[0.]*len(prefix)+[weight if f else 1. for f in focus],'time_focus':[False]*len(prefix)+focus})
        def __len__(self):return len(self.items)
        def __getitem__(self,index):return self.items[index]
    def collate(items):
        if len(items)!=1:raise ValueError('Temporal objective requires batch1')
        r=items[0]
        return {'input_ids':torch.tensor([r['input_ids']]),'attention_mask':torch.ones(1,len(r['input_ids']),dtype=torch.long),'labels':torch.tensor([r['labels']]),'loss_weights':torch.tensor([r['loss_weights']],dtype=torch.float32),'time_focus':torch.tensor([r['time_focus']],dtype=torch.bool)}
    class TemporalTrainer(CompletionTrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.field_diagnostics={}
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels=inputs.pop('labels');weights=inputs.pop('loss_weights');focus=inputs.pop('time_focus');valid=labels.ne(-100)
            n=int(valid.sum().item())
            if n<1 or not valid[:,-n:].all() or valid[:,:-n].any() or weights[~valid].ne(0).any() or focus[~valid].any():raise ValueError('Noncontiguous completion supervision')
            outputs=model(**inputs,logits_to_keep=n+1,use_cache=False)
            if outputs.logits.shape[1]!=n+1:raise ValueError('Shifted logits length mismatch')
            loss,stats=weighted_token_loss(outputs.logits[:,:-1,:],labels[:,-n:],weights[:,-n:],focus[:,-n:])
            for k,v in stats.items():self.field_diagnostics[k]=self.field_diagnostics.get(k,0)+v
            return (loss,outputs) if return_outputs else loss
    return TemporalDataset,collate,TemporalTrainer
