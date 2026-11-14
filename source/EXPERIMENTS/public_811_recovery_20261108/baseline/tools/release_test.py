"""Calibrate runtime parity, then run public inference; never train on test data."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

def run(label,cmd,p):
 with (p/(label+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTORCH_ALLOC_CONF':'expandable_segments:True'})
  for line in proc.stdout:log.write(line);log.flush();print(label+': '+line,end='',flush=True)
  rc=proc.wait()
 if rc:raise RuntimeError(label+' failed; inspect stage log')

def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--train-root',required=True);pa.add_argument('--public-root',required=True);pa.add_argument('--run-public',action='store_true');a=pa.parse_args()
 p=Path(a.project);tools=Path(__file__).resolve().parent;app=p/'app';root=Path(a.train_root)
 assert (p/'UPGRADE_COMPLETED.json').is_file(),'Complete train/base comparison/audit first'
 selection=json.loads((p/'selection.json').read_text());variant=selection['variant']
 refout=p/('calibration_'+variant);reference=rows(refout/'submission.jsonl');refeval=json.loads((refout/'evaluation.json').read_text());refplans={r['id']:r['plan'] for r in rows(refout/'plans.jsonl')}
 assert refeval['correct']+refeval['wrong']>0,'Zero answer coverage on calibration; diagnose plan format before a full public run'
 comparisons={};chosen=None
 for batch in [4,1]:
  out=p/('runtime_calibration_b'+str(batch))
  if not (out/'evaluation.json').is_file():
   run('RUNTIME_CAL_B'+str(batch),[sys.executable,'-u',str(tools/'infer_batched.py'),'--project',str(p),'--root',str(root),'--questions',str(p/'data/calibration_questions.jsonl'),'--variant',variant,'--output-dir',str(out),'--batch-size',str(batch),'--cache-dir',str(p.parent/'cache_033')],p)
   run('RUNTIME_SCORE_B'+str(batch),[sys.executable,'-u',str(app/'evaluate_saved.py'),'--root',str(root),'--questions',str(p/'data/calibration_questions.jsonl'),'--output-dir',str(out)],p)
  predicted=rows(out/'submission.jsonl');plans={r['id']:r['plan'] for r in rows(out/'plans.jsonl')};ev=json.loads((out/'evaluation.json').read_text());runtime=json.loads((out/'runtime.json').read_text())
  assert [r['id'] for r in predicted]==[r['id'] for r in reference]
  match=sum(r==s for r,s in zip(reference,predicted));planmatch=sum(plans.get(i)==v for i,v in refplans.items())
  comparisons[str(batch)]={'evaluation':ev,'runtime':runtime,'exact_output_matches':match,'n':len(reference),'exact_plan_matches':planmatch,'n_reference_plans':len(refplans),'reference_evaluation':refeval,'reference_runtime':json.loads((refout/'runtime.json').read_text()),'reference_seconds_exclude_load':True}
  (p/'runtime_comparison.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
  if match==len(reference) and ev['PA']>=refeval['PA'] and not runtime['network_attempts'] and not runtime['loader_errors']:
   chosen=batch;break
 assert chosen is not None,'Runtime differs on calibration; inspect evidence before a public run'
 runtime_selection={'model_variant':variant,'batch_size':chosen,'selected_using':'Calibration exact output parity plus PA; no audit/public labels used','comparisons':comparisons,'scope':'T4 runtime optimization; no 5090/Docker certification'}
 (p/'runtime_selection.json').write_text(json.dumps(runtime_selection,ensure_ascii=False,indent=2));print('RUNTIME_PARITY_VERIFIED:',json.dumps({'batch_size':chosen,'n':len(reference)},ensure_ascii=False),flush=True)
 if not a.run_public:return
 public=Path(a.public_root);assert public.name=='public_test' and (public/'questions.jsonl').is_file()
 out=p/('public_test_9b_b'+str(chosen)+'_frozen')
 if not (out/'format_audit.json').exists():
  run('PUBLIC_9B_BATCH',[sys.executable,'-u',str(tools/'infer_batched.py'),'--project',str(p),'--root',str(public),'--questions',str(public/'questions.jsonl'),'--variant',variant,'--output-dir',str(out),'--batch-size',str(chosen),'--cache-dir',str(p/'public_shop_cache')],p)
  run('AUDIT_PUBLIC_9B',[sys.executable,str(tools/'audit_output.py'),'--questions',str(public/'questions.jsonl'),'--submission',str(out/'submission.jsonl'),'--report',str(out/'format_audit.json')],p)
 audit=json.loads((out/'format_audit.json').read_text());assert audit['passed']
 result={'state':'completed','model_selection':selection,'runtime_selection':runtime_selection,'public_runtime':json.loads((out/'runtime.json').read_text()),'format_audit':audit,'output_dir':str(out),'public_PA':None,'public_gold_available':False,'submitted_to_organizer':False,'docker_tested':False,'public_used_for_training':False}
 (p/'PUBLIC_9B_VERIFIED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('PUBLIC_9B_VERIFIED:',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
