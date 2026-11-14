#!/bin/bash
export PYTHONIOENCODING=utf-8
~/zaic_khang_venv/bin/python /mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/tier1_eval.py \
  /mnt/d/ZAIC/EXPERIMENTS/khang_direction_v1/khang \
  /mnt/d/ZAIC/EXPERIMENTS/competition_ready_20261110/src/baseline/app \
  /mnt/d/ZAIC/PRIVATE_DATA/zaic/datamate_v1/qwen35_9b/data/split.json \
  /mnt/d/ZAIC/PRIVATE_DATA/khang_direction_v1/TIER1_PER_ID.jsonl
