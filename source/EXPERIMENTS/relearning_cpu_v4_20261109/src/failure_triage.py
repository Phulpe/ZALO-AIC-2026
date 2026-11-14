"""Do not turn an executor/scorer discrepancy into a model training target."""
from temporal_verifier import same_time

def plan_difference(predicted,target,as_of):
    if not isinstance(predicted,dict):return ['parse_or_schema']
    keys=set(predicted)|set(target);different=[]
    for key in sorted(keys):
        if key=='time':
            if not same_time(predicted.get(key),target.get(key),as_of):different.append(key)
        elif key=='order' and target.get('group_by') is None:continue
        elif predicted.get(key)!=target.get(key):different.append(key)
    return different

def classify(*,predicted,target,as_of,source_error=None,canonical_correct=False):
    if source_error:return {'class':'executor_or_source','train_eligible':False,'reason':source_error}
    if not canonical_correct:return {'class':'source_scorer_or_label_discrepancy','train_eligible':False,'reason':'canonical_plan_does_not_match_gold'}
    changed=plan_difference(predicted,target,as_of)
    if not changed:return {'class':'executor_repaired','train_eligible':False,'reason':'no_semantic_model_correction','regression_only':True}
    return {'class':'model_correction_verified','train_eligible':True,'changed_fields':changed}
