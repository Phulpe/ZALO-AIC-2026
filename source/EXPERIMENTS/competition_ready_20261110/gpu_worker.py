"""Separate GPU process for smoke or blind TRAIN regression; no public data."""
import sys, json, tempfile, math, time
from pathlib import Path


def main(stage, config_path):
    c = json.loads(Path(config_path).read_text())
    source = Path(c['source_root'])
    sys.path[:0] = [str(source), str(source/'baseline/app')]
    from relearning import atomic, warm_start, sha
    from permission import permitted
    if not permitted(c.get('competition_permission_evidence')):
        raise ValueError('Run permission not accepted')
    if not c['budget_approved'] or not c['gpu_confirmed']:
        raise ValueError('GPU budget missing')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One visible CUDA device required')
    from modeling import load_base, prepare_qlora
    from field_eval import verify_loaded
    from offline_guard import install
    install()
    with tempfile.TemporaryDirectory() as scratch:
        p = Path(scratch)
        (p/'assets').mkdir()
        (p/'assets/base').symlink_to(Path(c['base']), target_is_directory=True)
        if stage == 'smoke':
            from frozen_pilot import prepare
            _, _, selection = prepare(c)
            selected = json.loads((Path(c['frozen_regression_root'])/'PRIVATE_SELECTED.json').read_text())
            model, tok = load_base(p, True)
            model = prepare_qlora(model)
            model, parent = warm_start(model, c['adapter'], c)
            loaded = verify_loaded(model, c['adapter'])
            model.train()
            from time_objective import training_classes
            dataset, collate, trainer_class = training_classes(c['time_token_weight'])
            ds = dataset(selected, tok)
            # Longest selected fixture exercises the largest memory requirement.
            item = max(ds.items, key=lambda row: len(row['input_ids']))
            batch = {k: v.to('cuda:0') for k, v in collate([item]).items()}
            labels = batch['labels']
            n = int(labels.ne(-100).sum())
            assert n > 0 and labels[:, :-n].eq(-100).all()
            assert labels[0, -1].item() == tok.eos_token_id
            assert torch.equal(labels[:, -n:], batch['input_ids'][:, -n:])
            # This method only needs field_diagnostics, not Trainer/optimizer state.
            trainer = trainer_class.__new__(trainer_class)
            trainer.field_diagnostics = {}
            started = time.monotonic()
            loss = trainer.compute_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite smoke loss')
            loss.backward()
            grads = [v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite or missing smoke gradients')
            receipt = {
                'state': 'passed', 'GPU_executed': True, 'optimizer_executed': False,
                'fingerprints': {k: c[k] for k in ('parent_sha256', 'selected_dataset_sha256', 'source_archive_sha256')},
                'loss': float(loss.detach()), 'finite_gradient_tensors': len(grads),
                'completion_mask_shift_eos_valid': True,
                'tokens': len(item['input_ids']), 'elapsed_seconds': time.monotonic()-started,
                'peak_VRAM_bytes': torch.cuda.max_memory_allocated(),
                'load': loaded, 'selection': selection,
            }
            atomic(c['smoke_receipt'], receipt)
        elif stage == 'regression':
            from peft import PeftModel
            from runtime_data import load_learning_pool
            from field_eval import TrustedShops
            from source_annotations import VerifiedLearningShops
            from protected_regression import evaluate_protected
            from quality_gate import digest
            from pilot_gate import learning_gate
            model, tok = load_base(p)
            model = PeftModel.from_pretrained(model, c['candidate'], is_trainable=False, local_files_only=True)
            model.eval()
            loaded = verify_loaded(model, c['candidate'])
            pool, split, gold = load_learning_pool(c['runtime_root'], c['data_manifest_sha256'], c['split_manifest_sha256'])
            assert gold.rows is None
            shops = TrustedShops(c['cache_root'], c['cache_manifest_sha256'], c['runtime_manifest_sha256'])
            shops = VerifiedLearningShops(shops, c['runtime_root'], c['data_manifest_sha256'], c['split_manifest_sha256'])
            bank = Path(c['frozen_regression_root'])
            ids = set(json.loads(Path(c['critical_ids_path']).read_text())['ids'])
            selected = json.loads((bank/'PRIVATE_SELECTED.json').read_text())
            critical = [r for r in selected if r['kind'] == 'correction' and r['id'] in ids]
            assert len(critical) == 7
            started = time.monotonic()
            full, old7 = evaluate_protected(model, tok, bank, c['frozen_regression_manifest_sha256'],
                c['parent_sha256'], split['groups']['learning'], pool, gold, shops,
                Path(c['session_output'])/'regression', critical, digest(critical),
                stop_requested=lambda: time.monotonic()-started > c['regression_seconds'])
            atomic(Path(c['session_output'])/'REGRESSION_DECISION.json', {
                'full': full, 'old7': old7, 'loaded': loaded,
                'eligible_for_calibration': learning_gate(full)['passed'] and learning_gate(old7)['passed'],
                'promoted': False, 'public_ran': False,
            })
        else:
            raise ValueError('Unknown stage')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
