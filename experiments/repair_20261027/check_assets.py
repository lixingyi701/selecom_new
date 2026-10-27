"""CPU-only preflight. Never imports torch or starts a model."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import os
import sys

HERE = Path(__file__).resolve().parent

def main():
    root = Path(os.environ['SELECOM_ROOT']).expanduser().resolve()
    run = Path(os.environ['SELECOM_RUN_DIR']).expanduser().resolve()
    if run == root or run == root / 'research/repair-20261027':
        raise SystemExit('Choose a fresh experiment directory; original evidence is protected')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('Only physical GPU 0 is authorized')
    required = [
        'baselineModel/Qwen3-Embedding-0.6B/config.json',
        'baselineModel/Qwen2.5-7B-Instruct/config.json',
        'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector/encoder.pt',
        'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector/projector.pt',
        'checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator/adapter_model.safetensors',
        'data/stage1/stage1_train_data.jsonl', 'data/nq/eval/nq_eval.jsonl',
        'results/stage1_continue/checkpoint-79/encoder.pt',
    ]
    missing = [name for name in required if not (root / name).is_file()]
    manifest = json.loads((HERE / 'source-manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = HERE / record['path']
        assert path.resolve().is_relative_to(HERE)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['published_sha256']:
            raise SystemExit('Published source hash mismatch: ' + record['path'])
    for path in (HERE / 'scripts').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    if Path('/proc/meminfo').exists():
        memory = {a.rstrip(':'): int(b) for a, b, *_ in
                  (line.split() for line in Path('/proc/meminfo').read_text().splitlines())}
        if memory['MemAvailable'] < 24 * 1024**2:
            raise SystemExit('Available system memory below 24 GiB')
    expected = json.loads((HERE / 'environment.json').read_text())['packages']
    actual = {}
    for package in expected:
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = None
    print(json.dumps({'missing_assets': missing, 'source_hashes_verified': len(manifest['files']),
                      'environment_matches': actual == expected, 'installed_packages': actual,
                      'physical_gpu': 0, 'model_loaded': False}, indent=2))
    if missing:
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
