"""Separate train-only evaluation process; labels never enter inference."""
from pathlib import Path
from collections import defaultdict
import argparse,json
from scoring import matches,summarize

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--root',required=True);pa.add_argument('--questions',required=True);pa.add_argument('--output-dir',required=True);a=pa.parse_args()
 root=Path(a.root);out=Path(a.output_dir);assert root.name=='train','Only train-derived development labels may be read'
 qs=[json.loads(l) for l in Path(a.questions).read_text().splitlines() if l.strip()]
 preds=[json.loads(l) for l in (out/'submission.jsonl').read_text().splitlines() if l.strip()]
 assert [q['id'] for q in qs]==[r['id'] for r in preds]
 gold={r['id']:r for r in map(json.loads,(root/'answers.jsonl').read_text(encoding='utf-8-sig').splitlines()) if r}
 rows=[]
 for q,p in zip(qs,preds):
  state='abstain' if p['type']=='abstain' else 'correct' if matches(p,gold[q['id']]) else 'wrong'
  rows.append({'id':q['id'],'business_id':q['business_id'],'gold_type':gold[q['id']]['type'],'state':state})
 result=summarize(rows);result.update(by_business={biz:summarize([r for r in rows if r['business_id']==biz]) for biz in sorted({r['business_id'] for r in rows})},official_test=False,note='Business-disjoint train-derived development sample, separate calibration and audit. Small sample; not a public-test score.')
 (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print('DEVELOPMENT_EVALUATION:',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
