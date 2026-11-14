# ZAIC DataMate — Qwen3.5-9B repair v2.1

The public run is complete: 1,000 unique ordered predictions, 1,000 plan records and question logs, format audit passed. There are 235 non-abstentions and 765 abstentions. Public PA is unknown; nothing was submitted to the organizer. PA=1 has not been achieved.

See WORKLOG_VI.md and AGGREGATE_RESULTS.json for measured results, hashes, recovery provenance and limitations. This archive contains source code and aggregate evidence only, without competition questions, per-question answers, raw business data or weights. Source: challenge.zalo.ai.

## Repair and validation

v2 adds strict time/semantic contracts, relevant whole-rule retrieval, coverage handling, precise quantities and guarded execution. v2.1 fixes an invoice/catalog substring collision and invoice ID joins. The TRAIN regression recovered 95,845 dated lines and 51,044 orders. No business/question-ID branches or weight retraining were added.

99 tests passed (75 app + 24 tools) locally and on restored Colab. Calibration: 91 correct, 21 wrong, 88 abstain, PA .35 / 200. Fresh confirmation: 36 correct, 6 wrong, 58 abstain, PA .30 / 100. Historical business exposure is disclosed. Public labels were never used.

The adapter SHA256 is e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d.
Submission SHA256 is e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780.

## Source layout

- app/ and tools/: 27 frozen Python files. SOURCE_SHA256.json records their bytes.
- resume_public_v21.py: experiment-specific source/input/dependency checks, journal resume and final audit.
- run_public_local_with_checkpoints.py: separate Colab parent that saves immutable, checksum-verified Drive checkpoints and verified final files. Uses the user's existing authenticated account.
- versions.json and model_lock.json: recorded inference dependencies and model identity.

These recovery scripts contain experiment-specific Drive IDs/paths. They are Colab development infrastructure, not a Docker network-none entrypoint. Docker/RTX 5090 acceptance was not tested. Source ZIP is not a model image or weight archive.

## Reproducibility and safe continuation

Do not rerun all historical notebook cells. Inspect completion markers first. Run production inference on local storage; use separately verified immutable Drive snapshots. Shared Drive/FUSE overwrites produced duplicate filenames during recovery, so those interrupted folders are preserved for inspection.

Keep the exact frozen app/tools, model and adapter, public inputs, batch 4, input limit 3,072 and output limit 320. Runtime checks fingerprints before replaying plans. Never edit a fingerprint to force an incompatible checkpoint.

The completed local output is /content/zaic_v21_final_run/public_test_v21_frozen. The durable final result is https://drive.google.com/drive/folders/1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV. Colab /content is temporary. Durable checkpoints and metadata are under the user's ZAIC_9B_Resume_Own_20261107 Drive folder.

After code/config/weights changes, use a fresh experiment/output and a newly frozen TRAIN confirmation set. Existing confirmation 2 is now opened. Passing tests and a positive TRAIN PA do not prove a positive public PA.

## Tests

From the extracted source:
    cd app
    python -m unittest test_core.py test_v2_regression.py -v
    cd ..
    python -m unittest discover -s tools -p 'test*.py' -v

No per-question data is exported in this archive. Do not include raw competition data or credentials when sharing code.
