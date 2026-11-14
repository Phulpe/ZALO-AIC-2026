"""Offline deployment adapter; defaults to refusing an unapproved release."""
from pathlib import Path
import hashlib, json, os, shutil, sys, tempfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_release(root, manifest, validation=False):
    if manifest.get('ready_to_infer') is not True and not validation:
        raise ValueError('Release is not approved; do not deploy this draft')
    if manifest.get('model_selection') not in ('candidate', 'baseline'):
        raise ValueError('Explicit model selection required; no fallback')
    if not manifest.get('files'):
        raise ValueError('Complete release inventory required')
    for name, digest in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError('Packaged file mismatch')


def main():
    root = Path('/code')
    manifest = json.loads((root/'release_manifest.json').read_text())
    verify_release(root, manifest, os.environ.get('ZAIC_ALLOW_VALIDATION') == '1')
    questions = Path('/data/questions.jsonl')
    if not questions.is_file():
        raise ValueError('Expected read-only test mount /data/questions.jsonl')
    source = root/'src'
    sys.path[:0] = [str(source), str(source/'baseline/app')]
    # Use the same project structure and inference entry point as paired evaluation.
    with tempfile.TemporaryDirectory(prefix='zaic_infer_') as scratch:
        project = Path(scratch)
        (project/'assets').mkdir()
        (project/'assets/base').symlink_to(root/'models/base', target_is_directory=True)
        (project/'app').symlink_to(source/'baseline/app', target_is_directory=True)
        (project/'candidate').mkdir()
        (project/'candidate/final').symlink_to(root/'models/adapter', target_is_directory=True)
        shutil.copy2(source/'baseline/model_lock.json', project/'model_lock.json')
        import public_infer_811
        sys.argv = ['inference','--project',str(project),'--root','/data','--questions',str(questions),
            '--variant','candidate','--output-dir','/result','--batch-size','4',
            '--cache-dir','/tmp/zaic_cache','--max-input-tokens','3072']
        public_infer_811.main()


if __name__ == '__main__':
    main()
