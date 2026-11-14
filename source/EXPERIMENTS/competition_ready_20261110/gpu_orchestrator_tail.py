# Appended to a fingerprinted source/bootstrap generated locally.
import shutil, time
from relearning import verify_export, sha, atomic, seal
from isolated_training import run_stage

started = time.monotonic()
out = Path('/kaggle/working/competition_pilot')
out.mkdir()
try:
    # Reconstruct the immutable input without changing original Kaggle assets.
    markers = list(Path('/kaggle/input').rglob('TRANSFER_RECEIPT.json'))
    assert len(markers) == 1
    upload = json.loads(markers[0].read_text())
    bundle = Path('/kaggle/temp/frozen_input')
    bundle.mkdir(parents=True)
    archive = markers[0].parent/'FROZEN_V4.zip'
    if archive.is_file():
        assert sha(archive) == upload['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                assert (bundle/info.filename).resolve().is_relative_to(bundle.resolve())
                assert not stat.S_ISLNK(info.external_attr >> 16)
            z.extractall(bundle)
    else:
        manifests = [p for p in markers[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p) == upload['bundle_manifest_sha256']]
        assert len(manifests) == 1
        manifest = manifests[0]
        origin = manifest.parent.resolve()
        for name, item in json.loads(manifest.read_text())['files'].items():
            src = (origin/name).resolve()
            dst = (bundle/name).resolve()
            assert src.is_relative_to(origin) and dst.is_relative_to(bundle.resolve()) and not src.is_symlink()
            assert src.stat().st_size == item['bytes'] and sha(src) == item['sha256']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        shutil.copy2(manifest, bundle/'ARTIFACT_MANIFEST.json')
    assert verify_export(bundle) == CONFIG['bundle_manifest_sha256']
    bank = bundle/'BANK'
    assert verify_export(bank) == CONFIG['frozen_regression_manifest_sha256']
    bases = list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    runtimes = list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
    adapters = list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    caches = [p.parent for p in Path('/kaggle/input').rglob('CACHE_PROVENANCE.json') if p.parent.name == 'TRAIN_CACHE']
    assert len(bases) == len(runtimes) == len(adapters) == len(caches) == 1
    CONFIG.update(base=str(bases[0].parent/'base'), adapter=str(adapters[0].parent),
        runtime_root=str(runtimes[0].parent), cache_root=str(caches[0]),
        frozen_regression_root=str(bank), critical_ids_path=str(bundle/'CRITICAL_IDS.json'),
        source_root=str(root), session_output=str(out), output_root=str(out/'model_runs'),
        smoke_receipt=str(out/'GPU_SMOKE.json'), gpu_confirmed=True)
    assert sha(runtimes[0]) == CONFIG['runtime_manifest_sha256']
    assert verify_export(caches[0]) == CONFIG['cache_manifest_sha256']
    from frozen_pilot import prepare
    prepare(CONFIG)
    atomic(out/'RUN_CONFIG.json', CONFIG)
    request = out/'REQUEST.json'
    atomic(request, CONFIG)
    worker = Path('/kaggle/working/gpu_worker.py')
    worker.write_bytes(base64.b64decode(WORKER_PAYLOAD))
    assert sha(worker) == WORKER_HASH
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
    def stage(name, command, timeout):
        atomic(out/'PHASE.json', {'phase': name, 'elapsed_seconds': time.monotonic()-started})
        print('PHASE', name, flush=True)
        return run_stage(command, out/(name+'_process'), timeout, env)
    stage('smoke', [sys.executable, str(worker), 'smoke', str(request)], 900)
    # A completed smoke child has exited before loading the optimizer child.
    stage('train', [sys.executable, str(root/'frozen_pilot.py'), str(request), '--execute'], 1200)
    receipt = json.loads((out/'model_runs/FROZEN_PILOT_RECEIPT.json').read_text())
    run = Path(receipt['run_path'])
    assert verify_export(run) == receipt['manifest_sha256']
    CONFIG['candidate'] = str(run/'checkpoint')
    atomic(request, CONFIG)
    stage('regression', [sys.executable, str(worker), 'regression', str(request)], 3300)
    decision = json.loads((out/'REGRESSION_DECISION.json').read_text())
    atomic(out/'WORKFLOW_COMPLETED.json', {
        'state': 'completed', 'candidate_sha256': sha(run/'checkpoint/adapter_model.safetensors'),
        'run_id': run.name, 'regression': decision,
        'elapsed_seconds': time.monotonic()-started, 'promoted': False, 'public_ran': False,
    })
    print('PILOT_COMPLETE', json.dumps(decision), flush=True)
except Exception as exc:
    atomic(out/'FAILURE.json', {'exception_type': type(exc).__name__,
        'elapsed_seconds': time.monotonic()-started, 'partial_artifacts_preserved': True})
    raise
finally:
    seal(out)
