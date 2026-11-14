"""Read-only bounded official log stream; detailed events remain PRIVATE_DATA."""
from pathlib import Path
import subprocess, sys, json, re


def safe_line(value):
    line=value.strip()
    if re.fullmatch(r'BATCH_PROGRESS:\s+\d+\s+/\s+\d+\s+SECONDS\s+[\d.]+',line):return line
    if re.fullmatch(r'CALIBRATION_PHASE (parent|candidate)',line):return line
    for prefix in ('PAIRED_RECOVERY_RESULT','PAIRED_CALIBRATION_RESULT'):
        if line.startswith(prefix+' '):
            try:data=json.loads(line[len(prefix):])
            except ValueError:return None
            result={}
            for key in ('before','after'):
                if isinstance(data.get(key),dict):
                    result[key]={k:v for k,v in data[key].items() if k in ('n','correct','wrong','abstain','metric') and type(v) in (int,float)}
            for key in ('decision',):
                if isinstance(data.get(key),dict):
                    result[key]={k:v for k,v in data[key].items() if k in ('promote','correct_to_wrong','metric_delta') and type(v) in (bool,int,float)}
            for key in ('saved_plan_records','new_model_generations','fingerprint_unchanged'):
                if type(data.get(key)) in (int,bool):result[key]=data[key]
            return prefix+' '+json.dumps(result)
    return None


def main(kernel):
    if not re.fullmatch(r'lnghongphc/[a-z0-9-]+/\d+', kernel):
        raise ValueError('Explicit owned saved version required')
    private = Path(r'D:\ZAIC\PRIVATE_DATA\competition_ready_20261110')
    path = private/('LIVE_'+kernel.replace('/','_')+'.jsonl')
    script = '''from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import json,sys
sys.path.insert(0,ROOT)
from capture_live_phase import safe_line
api=KaggleApi();api.authenticate()
with Path(PATH).open('w',encoding='utf-8') as log:
 for event in api.kernels_logs_stream(KERNEL):
  log.write(json.dumps(event)+'\\n');log.flush()
  line=safe_line(event.get('data',''))
  if line:print(line,flush=True)
'''.replace('ROOT',repr(str(Path(__file__).parent))).replace('PATH',repr(str(path))).replace('KERNEL',repr(kernel))
    try:
        subprocess.run([sys.executable,'-X','utf8','-c',script],timeout=25,check=True)
    except subprocess.TimeoutExpired:
        print('Closed read-only log capture after25seconds; remote run untouched')


if __name__=='__main__':main(sys.argv[1])
