"""Optional calibration-fitted abstention policy; never changes answer values."""
import json
from pathlib import Path


def signature(plan,body):
    return '|'.join(str(x) for x in [plan.get('metric'),plan.get('group_by'),body.get('type')])


def load_policy(path):
    path=Path(path)
    if not path.exists():return None
    obj=json.loads(path.read_text())
    if obj.get('version')!='calibrated-abstention-v1' or obj.get('public_used') is not False:
        raise ValueError('Unrecognized abstention policy')
    if not isinstance(obj.get('allowed_signatures'),list) or not all(isinstance(x,str) for x in obj['allowed_signatures']):
        raise ValueError('Malformed policy signature list')
    return obj


def apply_policy(plan,body,policy):
    if policy is None or body['type'] in {'abstain','unanswerable'}:return body
    if signature(plan,body) in policy['allowed_signatures']:return body
    return {'type':'abstain','reason':'calibration_policy_reject'}
