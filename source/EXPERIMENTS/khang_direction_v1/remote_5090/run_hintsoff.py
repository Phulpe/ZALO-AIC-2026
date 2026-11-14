"""Chay datamate.runner voi hint bi an dong 'Code gợi ý' (giong cau khong co hint): LLM tu giai de kiem cheo hint.
Cac dong nhan dien khac (ky, ten khop...) giu nguyen nhu dinh dang SFT/build_teacher_sft. Tham so giong datamate.runner.
Dung (trong code/, PYTHONPATH=src): python3 run_hintsoff.py --data ... --out ... [tham so runner]
"""
import re
import sys

from datamate import runner as R

_CODE = re.compile(r"^Code gợi ý \([^)]*\): .*$\n?", re.M)
_orig = R.question_hints


def _no_code(*a, **k):
    return _CODE.sub("", _orig(*a, **k))


R.question_hints = _no_code
if __name__ == "__main__":
    sys.argv[0] = "datamate.runner"
    sys.exit(R.main())
