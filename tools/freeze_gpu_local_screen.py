"""Freeze required GPU-local worker placement with the unchanged 4b0e APP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from gpu_local_worker import policy
from descriptor_row_policy import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-package', type=Path, required=True)
    parser.add_argument('--worker-policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']):
        raise RuntimeError('freeze requires a clean canonical checkpoint')
    if subprocess.check_output(['git', '-C', str(source), 'diff', 'c9284c50', 'HEAD', '--', 'src/tensorfold']):
        raise RuntimeError('this placement screen requires unchanged c928 APP source')

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    policy(args.worker_policy, sha(args.worker_policy))
    shutil.copytree(args.parent_package, args.out)
    for name in ('gpu_local_worker.py', 'v100_backend_probe.py'):
        shutil.copy(source / 'tools' / name, args.out / 'tools' / name)
    shutil.copy(args.worker_policy, args.out / 'worker-affinity-policy.json')
    candidate = json.loads((args.out / 'candidate.json').read_text())
    candidate.update(worker_affinity='gpu-local', worker_affinity_policy_sha256=sha(args.worker_policy),
                     tool_sha256=sha(args.out / 'tools/v100_backend_probe.py'),
                     reference='c928/4b0e unbound actual screen; required GPU-local grouping is configuration, not optional on P3',
                     qualification='required whole GPU-local CPU mask, unchanged APP; actual all8 masks/resources/all6 unrun')
    for name in ('gpu_local_worker.py', 'v100_backend_probe.py'):
        candidate['tool_dependencies']['tools/' + name] = sha(args.out / 'tools' / name)
    (args.out / 'candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')
    binding = validate(args.out)
    controller = (args.parent_package / 'launch-finite.py').read_text()
    old = '/home/kklouzal/tensorfold-v100-qualification-20261026/perf85-descriptor-host-words-4b0e-workflow-v2-20261028'
    remote = '/home/kklouzal/tensorfold-v100-qualification-20261026/perf85-gpu-local-4b0e-20261028'
    if controller.count(old) != 1:
        raise ValueError('unexpected held parent controller output identity')
    controller = controller.replace(old, remote).replace('tensorfold-v100-descriptor-host-words-4b0e-workflow-v2',
                                                        'tensorfold-v100-gpu-local-4b0e')
    marker = "args += ['--overlap-content-capture'"
    if controller.count(marker) != 1:
        raise ValueError('unexpected held parent controller argument boundary')
    controller = controller.replace(marker,
        "args += ['--worker-affinity', candidate['worker_affinity'], '--worker-affinity-policy', '/qualification/worker-affinity-policy.json', '--worker-affinity-policy-sha', candidate['worker_affinity_policy_sha256']]\n" + marker)
    (args.out / 'launch-finite.py').write_text(controller)
    paths = ['tools/gpu_local_worker.py', 'tools/v100_backend_probe.py', 'tools/freeze_gpu_local_screen.py',
             'tests/test_gpu_local_worker.py', 'deploy/v100/worker-affinity.md', 'README.md']
    hold = {'schema': 'tensorfold.gpu-local-affinity-source-HOLD-v1',
            'commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_paths': {name: sha(source / name) for name in paths}, 'APP': candidate['wheel_sha256'],
            'APP_source_HOLD': candidate['source_hold_sha256'], 'row_policy_binding': binding,
            'worker_policy_sha256': candidate['worker_affinity_policy_sha256'],
            'parent_manifest_sha256': sha(args.parent_package / 'stage-files.json'),
            'controller_changes': 'fresh output/container + explicit affinity args only; backend measured loops unchanged',
            'configuration': 'Required per-rank GPU-local socket affinity; parent/quota/memory policy/math unchanged',
            'qualification': 'CPU controls separate; Root actual all8 mask/resource/all6 verification still required'}
    (args.out / 'source-hold-affinity.json').write_text(json.dumps(hold, indent=2) + '\n')
    files = {str(p.relative_to(args.out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'stage-files.json'}
    (args.out / 'stage-files.json').write_text(json.dumps(files, indent=2) + '\n')
    with tarfile.open(args.out.parent / 'payload.tar', 'w') as archive:
        for path in sorted(args.out.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(args.out)))
    ready = {'remote': remote, 'command': 'python3 ' + remote + '/launch-finite.py', 'APP': candidate['wheel_sha256'],
             'APP_HOLD': candidate['source_hold_sha256'], 'affinity_HOLD': sha(args.out / 'source-hold-affinity.json'),
             'worker_policy': candidate['worker_affinity_policy_sha256'], 'controller': sha(args.out / 'launch-finite.py'),
             'files': len(files), 'manifest': sha(args.out / 'stage-files.json'), 'payload_sha': sha(args.out.parent / 'payload.tar'),
             'GPU': 'UNRUN Root only; selected-file CPU preflight and sole D13 required first'}
    (args.out.parent / 'stage-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready, indent=2))


if __name__ == '__main__':
    main()
