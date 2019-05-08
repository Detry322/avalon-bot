import subprocess
import json
import os

DEEPROLE_BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'deeprole')
DEEPROLE_BINARY = os.path.join(DEEPROLE_BASE_DIR, 'code', 'deeprole')

deeprole_cache = {}

def actually_run_deeprole_on_node(node):
    command = [
        DEEPROLE_BINARY,
        '--play',
        '--proposer={}'.format(node['proposer']),
        '--succeeds={}'.format(node['succeeds']),
        '--fails={}'.format(node['fails']),
        '--propose_count={}'.format(node['propose_count']),
        '--depth=1',
        '--iterations=100',
        '--witers=50',
        '--modeldir=deeprole_models'
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEEPROLE_BASE_DIR
    )

    stdout, _ = process.communicate(input=str(node['new_belief']) + "\n")
    result = json.loads(stdout)
    return result


def run_deeprole_on_node(node):
    global deeprole_cache

    if len(deeprole_cache) > 100:
        deeprole_cache = {}

    cache_key = (node['proposer'], node['succeeds'], node['fails'], node['propose_count'], tuple(node['new_belief']))
    if cache_key in deeprole_cache:
        return deeprole_cache[cache_key]

    result = actually_run_deeprole_on_node(node)
    deeprole_cache[cache_key] = result
    return result
