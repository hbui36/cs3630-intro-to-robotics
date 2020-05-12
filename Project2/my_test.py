import numpy as np
import json
from Jinchen import *
from time import time


OBSERVATIONS = [i for i in range(10)]  # all possible vertical coordinates
STATES = [(i, j) for i in range(10) for j in range(10)]  # all 100 states
ACTIONS = ["Left", "Right", "Up", "Down"]  # the four possible actions


def generate_random_float_sys():
    global _rand_idx
    global _N
    ret = _pseudorandom_list[_rand_idx % _N]
    _rand_idx = _rand_idx + 1
    return ret


def normalize(nparr):
    return nparr / np.sum(np.array(nparr))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def sample_from_dbn_general(n):
  """Sample from a dynamic bayes network
  returns:
      samples (dict): A dictionary with the states, observations and actions sampled
  """
  samples = {'states': [], 'observations': [], 'actions': []}
  # Your code below

  s1 = sample_from_pmf(state_prior, STATES)
  o1 = sample_from_sensor_model(s1)
  samples['states'].append(s1)
  samples['observations'].append(o1)
  if n == 1:
    return samples
  
  s = s1
  for _ in range(n - 1):
    a = sample_from_pmf(action_prior, ACTIONS)
    s = sample_from_transition_model(s, a)
    o = sample_from_sensor_model(s)
    samples['actions'].append(a)
    samples['states'].append(s)
    samples['observations'].append(o)

  return samples


def test_action_prior():
    ret = []
    for test_param in ACTIONS:
        ret.append({
            'param': test_param,
            'result': action_prior(test_param)
        })
    return ret


def test_state_prior():
    ret = []
    for state in STATES:
        ret.append({
            'param': state,
            'result': state_prior(state)
        })
    return ret


def test_sensor_model():
    ret = []
    for state in STATES:
        for k in range(10):
            ret.append({
                'param': (k, state),
                'result': sensor_model(k, state)
            })
    return ret


def test_transition_model():
    ret = []
    for S in STATES:
        for T in STATES:
            for a in ACTIONS:
                ret.append({
                    'param': (T, S, a),
                    'result': '%.3f' % transition_model(T, S, a)
                })
    return ret


def test_transition_model_portal():
    ret = []
    for S in STATES:
        for T in STATES:
            for a in ACTIONS:
                ret.append({
                    'param': (T, S, a),
                    'result': '%.3f' % transition_model_portal(T, S, a)
                })
    return ret


def test_calculate_cdf():
    ret = []
    num_outcomes = [0, 1, 2, 5, 10, 20, 50, 100]
    for num_outcome in num_outcomes:
        ordered_outcome = [str(i) for i in range(num_outcome)]
        ordered_values = normalize([generate_random_float_sys() for _ in range(num_outcome)])
        table = dict(zip(ordered_outcome, ordered_values))
        def pmf(outcome):
            return table[outcome]
        cdf = calculate_cdf(pmf, ordered_outcome)
        curr_test_result = {}
        curr_test_result['size'] = num_outcome
        curr_test_result['function'] = {}
        for outcome in ordered_outcome:
            curr_test_result['function'][outcome] = '%.5f' % cdf(outcome)
        ret.append(curr_test_result)
    return ret


def test_sample_from_pmf():
    ret = []
    num_outcomes = [0, 1, 2, 5, 10, 20, 50, 100]
    sample_size = 100
    for num_outcome in num_outcomes:
        ordered_outcome = [str(i) for i in range(num_outcome)]
        ordered_values = normalize([generate_random_float_sys() for _ in range(num_outcome)])
        table = dict(zip(ordered_outcome, ordered_values))
        def pmf(outcome):
            return table[outcome]
        cdf = calculate_cdf(pmf, ordered_outcome)
        curr_test_result = {}
        curr_test_result['size'] = num_outcome
        curr_test_result['samples'] = []
        for _ in range(sample_size):
            curr_test_result['samples'].append(sample_from_pmf(pmf, ordered_outcome))
        ret.append(curr_test_result)
    return ret


def test_sample_from_sensor_model():
    sample_size_each_state = 100
    ret = []
    for state in STATES:
        for _ in range(sample_size_each_state):
            ret.append({
                'state': state,
                'result': sample_from_sensor_model(state)
            })
    return ret


def test_sample_from_transition_model():
    sample_size_each_state_action = 100
    ret = []
    for state in STATES:
        for action in ACTIONS:
            for _ in range(sample_size_each_state_action):
                ret.append({
                    'state_action': (state, action),
                    'result': sample_from_transition_model(state, action)
                })
    return ret


def test_sample_from_dbn():
    sample_size = 100
    ret = []
    for _ in range(sample_size):
        ret.append(sample_from_dbn())
    return ret


def test_maximum_probable_explanation():
    test_cases = json.loads(open('q11-case.json').read())[0]['case']
    ret = []
    for test_case in test_cases:
        test_case['states'] = maximum_probable_explanation(test_case['actions'], test_case['observations'])
        ret.append(test_case)
    return ret


def test_general_maximum_probable_explanation():
    test_cases = json.loads(open('q12-case.json').read())[0]['case']
    ret = []
    for test_case in test_cases:
        test_case['states'] = general_maximum_probable_explanation(test_case['actions'], test_case['observations'])
        ret.append(test_case)
    return ret


if __name__ == '__main__':
    _N = 100000
    fh = open('pseudorand.json', 'r')
    _pseudorandom_list = json.loads(fh.read())
    fh.close()
    _rand_idx = 0
    result = []

    # start = time()
    # result.append({
    #     'test': 'test_action_prior()',
    #     'output': test_action_prior()
    # })
    # stop = time()
    # print('test_action_prior(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_state_prior()',
    #     'output': test_state_prior()
    # })
    # stop = time()
    # print('test_state_prior(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_sensor_model()',
    #     'output': test_sensor_model()
    # })
    # stop = time()
    # print('test_sensor_model(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_transition_model()',
    #     'output': test_transition_model()
    # })
    # stop = time()
    # print('test_transition_model(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_transition_model_portal()',
    #     'output': test_transition_model_portal()
    # })
    # stop = time()
    # print('test_transition_model_portal(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_calculate_cdf()',
    #     'output': test_calculate_cdf()
    # })
    # stop = time()
    # print('test_calculate_cdf(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_sample_from_pmf()',
    #     'output': test_sample_from_pmf()
    # })
    # stop = time()
    # print('test_sample_from_pmf(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_sample_from_sensor_model()',
    #     'output': test_sample_from_sensor_model()
    # })
    # stop = time()
    # print('test_sample_from_sensor_model(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_sample_from_transition_model()',
    #     'output': test_sample_from_transition_model()
    # })
    # stop = time()
    # print('test_sample_from_transition_model(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_sample_from_dbn()',
    #     'output': test_sample_from_dbn()
    # })
    # stop = time()
    # print('test_sample_from_dbn(): {} ms'.format((stop - start) * 1000))

    # start = time()
    # result.append({
    #     'test': 'test_maximum_probable_explanation()',
    #     'output': test_maximum_probable_explanation()
    # })
    # stop = time()
    # print('test_maximum_probable_explanation(): {} ms'.format((stop - start) * 1000))

    start = time()
    result.append({
        'test': 'test_general_maximum_probable_explanation()',
        'output': test_general_maximum_probable_explanation()
    })
    stop = time()
    print('test_general_maximum_probable_explanation(): {} ms'.format((stop - start) * 1000))

    fh = open('test_output.json', 'w')
    fh.write(json.dumps(result, cls=NpEncoder))