"""Freeze a fresh private screen packet with separate APP and row authorities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from descriptor_row_policy import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application-package', type=Path, required=True)
    parser.add_argument('--row-package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']):
        raise RuntimeError('freeze requires a clean canonical workflow checkpoint')
    shutil.copytree(args.application_package, args.out)
    for name, target in [('source-hold.json', 'row-policy.json'),
                         ('wheel-provenance.json', 'row-policy-wheel-provenance.json')]:
        shutil.copy(args.row_package / name, args.out / target)
    shutil.copy(source / 'tools/descriptor_row_policy.py', args.out / 'tools/descriptor_row_policy.py')
    shutil.copy(source / 'tools/launch_descriptor_host_words_screen.py', args.out / 'launch-finite.py')

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    candidate = json.loads((args.out / 'candidate.json').read_text())
    candidate.update(row_policy_file='row-policy.json', row_policy_sha256=sha(args.out / 'row-policy.json'),
                     row_policy_provenance_sha256=sha(args.out / 'row-policy-wheel-provenance.json'))
    candidate['tool_dependencies']['tools/descriptor_row_policy.py'] = sha(args.out / 'tools/descriptor_row_policy.py')
    (args.out / 'candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')
    binding = validate(args.out)
    paths = ['tools/descriptor_row_policy.py', 'tools/launch_descriptor_host_words_screen.py',
             'tools/freeze_descriptor_host_words_screen.py', 'tests/test_descriptor_row_policy.py']
    hold = {'schema': 'tensorfold.descriptor-row-workflow-HOLD-v1',
            'commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_paths': {name: sha(source / name) for name in paths}, 'policy_binding': binding,
            'APP': candidate['wheel_sha256'], 'scope': 'Tool-only successor; no APP or GPU rerun',
            'original_packet': str(args.application_package),
            'original_manifest_sha256': sha(args.application_package / 'stage-files.json')}
    (args.out / 'source-hold-workflow.json').write_text(json.dumps(hold, indent=2) + '\n')
    files = {str(p.relative_to(args.out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'stage-files.json'}
    (args.out / 'stage-files.json').write_text(json.dumps(files, indent=2) + '\n')
    print(json.dumps(hold, indent=2))


if __name__ == '__main__':
    main()
