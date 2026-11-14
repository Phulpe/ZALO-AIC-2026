#!/bin/bash
# Test khang_v5: chay tu thu muc khang_v4 (co data/train), nap src + tests cua khang_v5. Thu muc tam trong ZAIC.
cd /mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/khang_v4
export PYTHONPATH=/mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/khang_v5/src PYTHONIOENCODING=utf-8
export TEST_TMP=/mnt/d/ZAIC/PRIVATE_DATA/khang_direction_v1/test_tmp TMPDIR=/mnt/d/ZAIC/PRIVATE_DATA/khang_direction_v1/test_tmp
mkdir -p $TEST_TMP/datamate_cache_dev; [ -L /tmp/datamate_cache_dev ] || { rm -rf /tmp/datamate_cache_dev; ln -s $TEST_TMP/datamate_cache_dev /tmp/datamate_cache_dev; }
for t in /mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/khang_v5/tests/test_*.py; do
  s=$(date +%s); ~/zaic_khang_venv/bin/python "$t" > $TEST_TMP/t.log 2>&1; rc=$?
  echo "$(basename $t) exit=$rc $(( $(date +%s)-s ))s :: $(tail -1 $TEST_TMP/t.log | cut -c1-120)"
done
