import subprocess
import json
import os
import numpy as np

from battlefield.bots.deeprole.lookup_tables import ASSIGNMENT_TO_VIEWPOINT

DEEPROLE_BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'deeprole')
DEEPROLE_BINARY = os.path.join(DEEPROLE_BASE_DIR, 'code', 'deeprole')

def marginalize_belief(belief):
    player_beliefs = [ np.zeros(15) for _ in range(5) ]

    for value, viewpoint in zip(belief, ASSIGNMENT_TO_VIEWPOINT):
        for player in range(5):
            player_beliefs[player][viewpoint[player]] += value

    for player in range(5):
        player_beliefs[player] /= np.sum(player_beliefs[player])

    result = np.zeros(60)

    for index, viewpoint in enumerate(ASSIGNMENT_TO_VIEWPOINT):
        for player in range(5):
            result[index] += np.log(player_beliefs[player][viewpoint[player]])

    result -= np.max(result)
    result = np.exp(result)
    result /= np.sum(result)

    nonzero = np.array([1.0 if b != 0 else 0.0 for b in belief])
    nonzero /= np.sum(nonzero)

    result = (1-1e-60) * result + 1e-60 * nonzero

    return list(result / np.sum(result))


deeprole_cache = {}

def actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder):
    command = [
        DEEPROLE_BINARY,
        '--play',
        '--proposer={}'.format(node['proposer']),
        '--succeeds={}'.format(node['succeeds']),
        '--fails={}'.format(node['fails']),
        '--propose_count={}'.format(node['propose_count']),
        '--depth=1',
        '--iterations={}'.format(iterations),
        '--witers={}'.format(wait_iterations),
        '--modeldir={}'.format(nn_folder)
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEEPROLE_BASE_DIR
    )

    if no_zero:
        belief = node['nozero_belief']
    else:
        belief = node['new_belief']

    stdout, _ = process.communicate(input=str(belief) + "\n")
    result = json.loads(stdout)
    return result


def run_deeprole_on_node(node, iterations, wait_iterations, no_zero=False, nn_folder='deeprole_models'):
    global deeprole_cache

    if len(deeprole_cache) > 250:
        deeprole_cache = {}

    cache_key = (
        node['proposer'],
        node['succeeds'],
        node['fails'],
        node['propose_count'],
        tuple(node['new_belief']),
        iterations,
        wait_iterations,
        no_zero,
        nn_folder
    )
    
    if cache_key in deeprole_cache:
        return deeprole_cache[cache_key]

    result = actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder)
    deeprole_cache[cache_key] = result
    return result
