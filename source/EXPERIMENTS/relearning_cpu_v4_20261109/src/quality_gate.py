"""Fail-closed supervision selection, provenance and frozen evaluation gates."""
import hashlib,json,collections,math

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def select(rows,evidence,config):
    if not rows:raise ValueError('Empty bank')
    parent=config['parent_sha256'];allowed=set(config['learning_businesses'])
    blocked=set(config['calibration_businesses'])|set(config['holdout_businesses'])
    if allowed & blocked:raise ValueError('Split overlap')
    if type(config['max_per_kind']) is not int or config['max_per_kind']<1:raise ValueError('Selection budget')
    policy=config.get('selection_policy','business_metric_v1')
    if policy not in {'business_metric_v1','time_business_metric_v1'}:raise ValueError('Unknown selection policy')
    temporal=policy=='time_business_metric_v1'
    minimum_time=config.get('min_replay_time_buckets',1)
    if temporal and (type(minimum_time) is not int or minimum_time<1):raise ValueError('Temporal diversity requirement')
    groups={'correction':collections.defaultdict(list),'replay':collections.defaultdict(list)}
    seen_ids=set();seen_prompts={}
    for row in sorted(rows,key=lambda x:x['id']):
        if row['id'] in seen_ids:raise ValueError('Duplicate ID')
        seen_ids.add(row['id'])
        if row['split']!='learning' or row['business_id'] not in allowed or row['business_id'] in blocked:raise ValueError('Wrong split')
        if row['parent_sha256']!=parent or row['target_verified'] is not True:raise ValueError('Wrong parent or unverified')
        kind=row['kind']
        if kind not in groups:raise ValueError('Unknown target role')
        plan=json.loads(row['response']);prompt=row['prompt'];ev=evidence[row['id']]
        required={'schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction'}
        if any(ev.get(k) is not True for k in required):raise ValueError('Incomplete target evidence')
        for key in ('source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256'):
            if ev.get(key)!=config[key]:raise ValueError('Evidence fingerprint mismatch')
        if ev.get('plan_sha256')!=digest(plan) or ev.get('prompt_sha256')!=digest(prompt):raise ValueError('Target changed after verification')
        if ev.get('parent_sha256')!=parent:raise ValueError('Prediction parent mismatch')
        if kind=='correction' and ev.get('first_pass_verified_correct') is not False:raise ValueError('Not an observed model failure')
        if kind=='replay' and ev.get('first_pass_verified_correct') is not True:raise ValueError('Not verified first-pass replay')
        key=digest(prompt)
        if key in seen_prompts:
            if seen_prompts[key]!=digest(plan):raise ValueError('Conflicting duplicate target')
            continue
        seen_prompts[key]=digest(plan)
        bucket=ev.get('temporal_bucket')
        if temporal and bucket not in {'month','day','range','today','yesterday','this_week','last_week','this_month','last_month','this_year','last_year','rolling_days','until','none'}:raise ValueError('Missing or unknown temporal evidence')
        groups[kind][(row['business_id'],plan['metric'],bucket) if temporal else (row['business_id'],plan['metric'])].append(row)
    def balanced(buckets):
        result=[];by_metric=collections.defaultdict(list)
        for key in sorted(buckets):by_metric[key[1]].append(key)
        while len(result)<config['max_per_kind'] and any(buckets.values()):
            for metric in sorted(by_metric):
                keys=by_metric[metric]
                while keys and not buckets[keys[0]]:keys.pop(0)
                if keys and len(result)<config['max_per_kind']:
                    key=keys.pop(0);result.append(buckets[key].pop(0))
                    if buckets[key]:keys.append(key)
        return result
    def balanced_time(buckets):
        result=[];business_counts=collections.Counter();metric_counts=collections.Counter();times=sorted({key[2] for key in buckets})
        while len(result)<config['max_per_kind'] and any(buckets.values()):
            for period in times:
                keys=[key for key in buckets if key[2]==period and buckets[key]]
                if not keys or len(result)>=config['max_per_kind']:continue
                key=min(keys,key=lambda k:(business_counts[k[0]],metric_counts[k[1]],k[0],k[1]));result.append(buckets[key].pop(0));business_counts[key[0]]+=1;metric_counts[key[1]]+=1
        return result
    choose=balanced_time if temporal else balanced
    correction=choose(groups['correction']);replay=choose(groups['replay']);n=min(len(correction),len(replay))
    if n<config['min_per_kind']:raise ValueError('Insufficient diverse verified supervision')
    selected=[row for pair in zip(correction[:n],replay[:n]) for row in pair]
    if len({r['business_id'] for r in selected})<config['min_businesses']:raise ValueError('Insufficient business diversity')
    report={'correction':n,'replay':n,'rows':len(selected),'unique_businesses':len({r['business_id'] for r in selected}),'dataset_sha256':digest(selected),'selection':'deterministic round-robin by business and metric; no oversampling'}
    if temporal:
        report.update(selection='round-robin time; least-used business/metric; no oversampling',replay_time_buckets=dict(collections.Counter(evidence[r['id']]['temporal_bucket'] for r in replay[:n])),correction_time_buckets=dict(collections.Counter(evidence[r['id']]['temporal_bucket'] for r in correction[:n])))
        if len(report['replay_time_buckets'])<minimum_time:raise ValueError('Insufficient temporal replay diversity')
    return selected,report

def promotion(before,after,config):
    if before['manifest_sha256']!=config['calibration_manifest_sha256'] or after['manifest_sha256']!=before['manifest_sha256']:raise ValueError('Evaluation split changed')
    if before['decoder_sha256']!=after['decoder_sha256'] or before['scorer_sha256']!=after['scorer_sha256']:raise ValueError('Evaluation implementation changed')
    if set(before['verdicts'])!=set(after['verdicts']) or not before['verdicts']:raise ValueError('Missing evaluation IDs')
    if any(type(v) is not bool for obj in (before,after) for v in obj['verdicts'].values()):raise ValueError('Invalid verdict')
    if any(not math.isfinite(obj['metric']) for obj in (before,after)):raise ValueError('Nonfinite score')
    lost=sum(before['verdicts'][k] and not after['verdicts'][k] for k in before['verdicts'])
    return {'promote':after.get('gates_passed') is True and lost<=config['max_regressions'] and after['metric']>=before['metric']+config['minimum_improvement'],'correct_to_wrong':lost,'metric_delta':after['metric']-before['metric']}
