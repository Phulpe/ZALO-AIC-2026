"""Aggregate tests; Windows portability issues are reported, never hidden."""
from pathlib import Path
import sys,unittest,json,io,hashlib,os
root=Path(__file__).parent
sys.path[:0]=[str(root),str(root/'baseline/app'),str(root/'baseline/tools')]
import relearning as r

def suite_result(names):
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.TestLoader().loadTestsFromNames(names))
    return {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()},stream.getvalue()

def main():
    before={str(p):r.sha(p) for p in (root/'baseline').rglob('*.py')}
    new,new_log=suite_result(['test_relearning','test_bridge','test_cycle','test_smoke_validation','test_permission','test_runtime_data','test_learning_probe','test_gpu_runtime','test_prediction_failures','test_workflow_811','test_public_recovery','test_loader_probe'])
    baseline,baseline_log=suite_result(['test_core','test_v2_regression','test_audit_output','test_batch_io','test_policy'])
    assert all(r.sha(Path(p))==h for p,h in before.items()),'Baseline changed'
    report={'new':new,'baseline':baseline,'baseline_python_bytes_unchanged':True,'platform':sys.platform,'gpu_executed':False}
    r.atomic(root/'TEST_RESULTS.json',report)
    (root/'TEST_DETAILS.txt').write_text(new_log+'\n'+baseline_log,encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if new['passed'] and baseline['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
