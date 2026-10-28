"""Finite sequential pre20/cost3/post20 runner; parent owns no CUDA context."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    out = Path('/evidence')
    env = dict(os.environ)
    env['PYTHONPATH'] = '/qualification/tests:/qualification/tests/cuda:/qualification'
    def run(argv, label):
        try:
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env=env, timeout=180)
            code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
        except subprocess.TimeoutExpired as error:
            code, stdout, stderr, timed_out = 124, error.stdout or b'', error.stderr or b'', True
        (out / (label + '.stdout')).write_bytes(stdout)
        (out / (label + '.stderr')).write_bytes(stderr)
        record = {'argv': argv, 'exit': code, 'timeout_seconds': 180, 'timed_out': timed_out}
        (out / (label + '-command.json')).write_text(json.dumps(record, indent=2) + '\n')
        return code
    numerical = [sys.executable, '-m', 'tools.dp4a_numeric_gate']
    pre = run(numerical + ['pre', '--out', str(out)], 'numeric-pre')
    if pre:
        raise SystemExit(pre)
    model = '/models/sc117/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
    cost = run([sys.executable, '-m', 'tools.dp4a_actual_cost', '--source', '/qualification',
                '--model', model, '--out', str(out / 'cost')], 'actual-cost')
    # A completed failed child leaves no live diagnostic owner in this parent;
    # still retain postcondition evidence without concealing the cost failure.
    post = run(numerical + ['post', '--out', str(out)], 'numeric-post')
    complete = pre == cost == post == 0
    if complete:
        before = json.loads((out / 'numeric-pre.json').read_text())
        after = json.loads((out / 'numeric-post.json').read_text())
        result = json.loads((out / 'cost/result.json').read_text())
        complete = (before['complete'] and after['complete'] and len(result['cases']) == 3
                    and before['module_sha256'] == after['module_sha256'] == result['native_module_sha256']
                    and all(case['status'] == [0] and not case['exact_only']
                            and case['bits_repeatable_within_each_path'] for case in result['cases']))
    (out / 'result.json').write_text(json.dumps({'complete': complete, 'pre_exit': pre, 'cost_exit': cost,
        'post_exit': post, 'scope': 'actual-bank routed-only CUDA cost diagnostic; no model/P3/quality/admission claim'}, indent=2) + '\n')
    raise SystemExit(0 if complete else cost or post or 1)


if __name__ == '__main__':
    main()
