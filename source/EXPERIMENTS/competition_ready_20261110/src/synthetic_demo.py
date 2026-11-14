"""Executed demonstration of the ledger/correction/replay loop, no real data."""
from pathlib import Path
import json
from relearning import new_run,LearningRound,mixture,atomic,seal

def main(output):
    cfg={'mode':'learning','init_mode':'warm_start','attempts_per_question':2,'learning_businesses':['synthetic'],'calibration_businesses':['cal'],'holdout_businesses':['hold'],'parent_sha256':'synthetic-test-only-no-model-loaded'}
    run=new_run(output,cfg)
    pool=[{'id':str(i),'business_id':'synthetic','split':'learning','prompt':'Synthetic sum 1+1','as_of_date':'2026-01-01'} for i in range(4)]
    def predict(q,n):
        if q['id']=='2':raise TimeoutError('synthetic failure')
        return {'operation':'sum' if q['id']=='0' else 'bad','value':2 if q['id'] in {'0','3'} else 0}
    def execute(q,p):return p['value']
    def semantic(q,p):return p['operation']=='sum'
    def correction(q,row,gold):
        if q['id']!='1':raise ValueError('No verified correction yet')
        return {'operation':'sum','value':2}
    metrics,replay,corrections=LearningRound(run,cfg,pool,lambda i:2,predict,execute,lambda p:None,semantic,lambda a,b:a==b,correction).process()
    dataset=mixture(replay,corrections,['synthetic'])
    atomic(run/'verified_dataset.json',dataset)
    metrics.update(synthetic_only=True,optimizer_executed=False,dataset_rows=len(dataset))
    atomic(run/'DEMO_RESULT.json',metrics);seal(run)
    print(json.dumps({'run_id':run.name,**metrics},indent=2));return run

if __name__=='__main__':main(Path(__file__).parent/'synthetic_runs')
