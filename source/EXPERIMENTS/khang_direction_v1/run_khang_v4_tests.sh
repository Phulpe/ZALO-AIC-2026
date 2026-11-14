#!/bin/bash
cd /mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/khang_v4
export PYTHONPATH=src PYTHONIOENCODING=utf-8
for t in tests/test_*.py; do
  s=$(date +%s); ~/zaic_khang_venv/bin/python "$t" > /tmp/t.log 2>&1; rc=$?
  echo "$t exit=$rc $(( $(date +%s)-s ))s :: $(tail -1 /tmp/t.log | cut -c1-120)"
done
