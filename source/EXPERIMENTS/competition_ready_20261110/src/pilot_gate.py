"""Fail-closed pilot selection before any full calibration/public stage."""
def learning_gate(summary):
    required=['expected','attempted','real_output_ids','correction_before_correct','correction_after_correct','correction_total','replay_before_correct','replay_after_correct','correct_to_wrong']
    if any(type(summary.get(k)) is not int or summary[k]<0 for k in required):raise ValueError('Invalid post-training counts')
    if summary['mode']!='evaluation' or summary['weights_updated'] is not False:raise ValueError('Evaluation changed weights/mode')
    if not summary['expected']==summary['attempted']==summary['real_output_ids']:raise ValueError('Missing real predictions')
    if summary['expected']!=summary['correction_total']+summary['replay_before_correct']:raise ValueError('Role counts do not reconcile')
    if summary['replay_after_correct']>summary['replay_before_correct'] or summary['correction_after_correct']>summary['correction_total']:raise ValueError('Impossible counts')
    if summary['correct_to_wrong']!=summary['replay_before_correct']-summary['replay_after_correct']:raise ValueError('Regression counts mismatch')
    return {'passed':summary['correction_after_correct']>summary['correction_before_correct'] and summary['correct_to_wrong']==0,'correction_gain':summary['correction_after_correct']-summary['correction_before_correct'],'correct_to_wrong':summary['correct_to_wrong'],'best_model_promoted':False}

def after_training(config,post_evaluate,calibrate=None,public=None):
    """Post-test is mandatory in this runtime, before optional full evaluation."""
    report=post_evaluate();gate=learning_gate(report)
    result={'learning_gate':gate,'calibration_ran':False,'public_ran':False,'promoted':False}
    if not gate['passed']:return result
    if config.get('run_full_calibration'):
        if calibrate is None:raise ValueError('Missing frozen calibration runner')
        decision=calibrate();result.update(calibration_ran=True,promoted=bool(decision.get('promote')))
        if result['promoted'] and config.get('run_public'):
            if public is None:raise ValueError('Missing public runner')
            public();result['public_ran']=True
    elif config.get('run_public'):raise ValueError('Public requires complete calibration')
    return result
