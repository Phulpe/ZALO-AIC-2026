from pathlib import Path
import os,sys,hashlib,zipfile,stat,json
wheels=list(Path('/kaggle/input').rglob('bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl'))
assert len(wheels)==1,'Need pinned official bitsandbytes wheel'
wheel=wheels[0]
assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283'
deps=Path('/kaggle/temp/zaic_bnb_0502');deps.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(wheel) as z:
    for info in z.infolist():
        assert (deps/info.filename).resolve().is_relative_to(deps.resolve())
        assert not stat.S_ISLNK(info.external_attr>>16)
    z.extractall(deps)
sys.path.insert(0,str(deps));os.environ['PYTHONPATH']=str(deps)+os.pathsep+os.environ.get('PYTHONPATH','')
import bitsandbytes as bnb
assert bnb.__version__=='0.50.2'
print('OFFICIAL_BNB_READY',bnb.__version__)
