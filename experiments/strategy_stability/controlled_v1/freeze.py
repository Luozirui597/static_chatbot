"""Hash frozen assets and the actual verification/integration implementation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    excluded = {'manifest.json'}
    result = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    result += [REPO / p for p in (
        'experiments/strategy_stability/verifier.py',
        'experiments/strategy_stability/integration_app.py',
        'experiments/strategy_stability/models.py',
        'backend/reply_language.py', 'backend/llm_client.py', 'backend/review_diagnostics.py',
    )]
    return sorted(result)


def write():
    manifest = {'version':'controlled-freeze-v1', 'files':{str(p.relative_to(REPO)):digest(p) for p in files()}}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    current = {str(p.relative_to(REPO)):digest(p) for p in files()}
    if manifest['files'] != current:
        different = sorted(k for k in set(current) | set(manifest['files']) if current.get(k) != manifest['files'].get(k))
        raise ValueError('frozen asset drift: ' + ', '.join(different))
    return digest(ROOT / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        write()
    print('manifest verified', verify())
