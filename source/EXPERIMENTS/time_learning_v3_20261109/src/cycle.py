"""One bounded correction/replay cycle; explicit runtime providers, no fallback."""
from pathlib import Path
from relearning import LearningRound,new_run,mixture,regressions,atomic,seal,track
from permission import permitted

def run_cycle(config,pool,gold_reader,predict,execute,validate,semantic,scorer,
              correct,evaluate,learner,protected=()):
    """evaluate is a locked evaluation runner, returns verdicts and frozen split hash.

    learner owns model lifetime and may call train_warm.train after releasing
    the parent inference model. Evaluation providers never receive learning gold.
    """
    if config.get('init_mode')!='warm_start' or config.get('mode')!='learning':raise ValueError('Explicit learning/warm_start required')
    if config.get('competition_assets') and not permitted(config.get('competition_permission_evidence')):raise ValueError('Competition processing permission not accepted')
    run=new_run(config['output_root'],config,protected)
    baseline=evaluate(config['parent_sha256'],mode='evaluation')
    required={'calibration_manifest_sha256','verdicts','metric'}
    if not required<=set(baseline):raise ValueError('Evaluation evidence incomplete')
    metrics,replay,correction=LearningRound(run,config,pool,gold_reader,predict,execute,validate,semantic,scorer,correct).process()
    atomic(run/'before_evaluation.json',baseline)
    if metrics['attempt_coverage']!=1:raise ValueError('Incomplete learning pool coverage')
    if not correction or not replay:
        atomic(run/'CYCLE_RESULT.json',{'status':'no_verified_balanced_dataset','training_started':False,'metrics':metrics});seal(run)
        return run
    dataset=mixture(replay,correction,config['learning_businesses'])
    atomic(run/'correction_replay_manifest.json',{'rows':len(dataset),'replay':len(replay),'correction':len(correction)})
    # Explicit callback; no path can silently instantiate a fresh adapter.
    candidate=Path(learner(config,dataset))
    after=evaluate(str(candidate),mode='evaluation')
    if after['calibration_manifest_sha256']!=baseline['calibration_manifest_sha256']:raise ValueError('Calibration changed between parent/candidate')
    lost=regressions(baseline['verdicts'],after['verdicts'])
    atomic(run/'after_evaluation.json',after)
    evidence={'calibration_manifest_sha256':after['calibration_manifest_sha256'],'candidate_metric':after['metric'],'minimum_metric':baseline['metric']+config['minimum_improvement'],'regression_count':len(lost),'max_regressions':config['max_regressions'],'gates_passed':after.get('gates_passed') is True}
    promoted=track(Path(config['output_root'])/'STATE.json',candidate,'promote',evidence)
    atomic(run/'CYCLE_RESULT.json',{'status':'evaluated','candidate':str(candidate),'correct_to_wrong':lost,'promoted':promoted,'metrics':metrics})
    seal(run);return run
