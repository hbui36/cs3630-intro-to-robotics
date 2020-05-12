# -*- coding: utf-8 -*-
"""Copy of robot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_qPgGpS-HKGtxEv4MexJCAj2Ov_jIvHo

# Project 2: Probabilistic Inference

Due Date: Monday, February 3, 2020 @ 11:59 P.M.

Student Name: ENTER NAME HERE

In this project, you will be implementing code to perform probabilistic inference in a grid world. You will also be asked to answer some open-ended questions as well. The submission details are provided at the end of the page.

### Instruction
For this project, you'll be using Google colab, which is a Jupyter Notebook hosted on Google Drive. If you've used Jupyter Notebook before, it'll feel familiar to you. If not, feel free to watch this short introductory video: https://youtu.be/inN8seMm7UI 

In order to start working on the assignment, you would need to save a copy of this colab onto your Google Drive as this file is read-only. This can be done by going on the top left hand corner and selecting `File -> Save a copy in Drive...` It should automatically create a copy and store a local copy on your Google Drive. Now you can edit the file! If you're curious on where the file is stored, select `File -> Locate in Drive`

Your job is to implement the functions below. Don't change the function names or parameters as they need to be consistent with the autograders. Make sure the output types are consistent with the comments as well.

# Coding [50 Points]

In the coding section, you will be implementing functions to calculate certain probability functions, sampling from these functions, as well as performing Bayes Net Inference. Please go through this first code section carefully, with the imports and variables you will need. Also, please use `generate_random_float` to get a random float between 0 and 1.
"""

# All imports and variables you will need

import numpy as np
OBSERVATIONS = [i for i in range(10)]  # all possible vertical coordinates
STATES = [(i, j) for i in range(10) for j in range(10)]  # all 100 states
ACTIONS = ["Left", "Right", "Up", "Down"]  # the four possible actions

def generate_random_float():
  """ Returns a random float between 0 and 1 
  """
  return np.random.rand()

"""## Calculating Probabilities

### 1. `action_prior(action)` [2 points]

This function will take in one of the actions in `[Left, Right, Up, Down]`, and return the probability of that action occurring. You will be returning a `float` between 0 and 1 which is the probability of that action. You will be using the following table as the probability mass function for that action.

$a_{i}$ | $P$$($$A$$=$$a_{i}$$)$
--- | ---
Left | 0.2
Right | 0.6
Up | 0.1
Down | 0.1
"""

def action_prior(action):
  """Return prior probability distribution on an action.
    arguments:
      action (string): "Left", "Right", "Up", "Down"
    returns:
      probability of given action (float).
  """
  dist = {
    "Left": 0.2,
    "Right": 0.6,
    "Up": 0.1,
    "Down": 0.1
  }
  # assert action in dist, 'action "{}" not in dist'.format(action)
  return dist[action]

"""### 2. `state_prior(state)` [2 points]

This function will take in one of the states of the grid world and return the probability of the robot being in that state. You will be returning a `float` between 0 and 1 which is the probability of that action. A state is represented by $(i,j)$, where $i$ is the row number in the grid, between 0 and 9 and $j$ is the column number in the grid, between 0 and 9. You will be using the following grid world diagram to calculate the probability of a state. You should use a numpy array to represent the grid world. The origin of this grid world (0,0) is in the top left corner. $i$ increases as you go down in the grid, and $j$ increases as you go right in the grid.

![](https://i.ibb.co/TqfCLCw/grid-world.png)
"""

def state_prior(state):
  """Return prior probability distribution on a state.
  arguments:
      state (tuple): (i: int, j: int)
  returns:
      probability of given state (float).
  """
  # Use np.array to create a 2D array that represents the prior on the state
  # (i,j) where i is the row index and j is the column index of the agent's
  # position.
  states_dist = np.array([
    [.0, .0, .0, .0, .1, .0, .0, .0, .0, .0,],
    [.0, .1, .0, .0, .0, .0, .0, .0, .0, .0,],
    [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0,],
    [.0, .0, .0, .1, .0, .0, .0, .0, .0, .0,],
    [.1, .0, .0, .0, .0, .1, .0, .0, .1, .0,],
    [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0,],
    [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0,],
    [.0, .0, .1, .0, .0, .0, .1, .0, .0, .0,],
    [.0, .0, .0, .0, .0, .0, .0, .1, .0, .0,],
    [.0, .0, .0, .0, .0, .0, .0, .0, .0, .1,]
  ])
  # assert type(state) is tuple and len(state) == 2 and type(state[0]) is int and type(state[1]) is int, 'invalid argument state'
  # assert state[0] >= 0 and state[1] >= 0 and state[0] < states_dist.shape[0] and state[1] < states_dist.shape[1], 'state out of bounds'
  return float(states_dist[[state[0]], [state[1]]])

"""### 3. `sensor_model(k, S)` [3 points]

This function will take in a value of the sensor $k$, and the state $(i,j)$ that the robot is at. You will need to calculate $P(O=k|S)$. The sensor reports the row that the robot is at in the grid world. You can use the following equations to implement this function.

![](https://i.ibb.co/qxF09cW/sensor-model.png)
"""

# Define the sensor model conditional distribution, P(O|S)
def sensor_model(k, S):
  """Give value of Conditional Probability Table (CPT) for sensor model P(O|S).
  arguments:
      k: the value of the sensor, corresponding to a row in the grid world (i).
      S (int,int): a tuple (i,j), representing the position in the grid.
  returns:
      P(k|i,j) (float).
  notes:
      Sensor reports noisy observation of vertical coordinate i.
  """
  if k == S[0]:
    return 0.91
  else:
    return 0.01

"""### 4.`transition_model(T, S, A)` [5 points]

This function will take in the next state the robot is in, $T = (i,j)$, the current state the robot is in, $S = (i,j)$, and $A$, an action from `[Left, Right, Up, Down]`. You will be calculating $P(T|S, A)$. You will be using the following statements to calculate the probability.


$T$ is the correct state when you take action $A$ from $S$; $P(T|S,A) = 0.85$ \\
$T$ is a possible state when you take an action other than $A$ from $S$; $P(T|S,A) = 0.05$ \\
Otherwise; $P(T|S,A) = 0$

This model accounts for the robot possibly taking the wrong action.

**NOTE:** If the robot is at the edge of the grid world, and tries to perform an action that will take it out of the grid world, it will stay in the same state (location).
"""

# Define the state transition model P(T|S,A)
def transition_model(T, S, A):
  """Give value of CPT for transition model P(T|S,A).
  arguments:
      T (int,int): a tuple (i,j), representing the next position in the grid
      S (int,int): a tuple (i,j), representing the position in the grid.
      A (str): a string representing the action to be taken
  returns:
      P(T|S,A) (float).
  """
  def next_state(S, A):
    transition_offset = {
      'Left': (0, -1),
      'Right': (0, 1),
      'Up': (-1, 0),
      'Down': (1, 0)
    }
    temp = (S[0] + transition_offset[A][0], S[1] + transition_offset[A][1])
    if temp in STATES:
      return temp
    else:
      return S
   
  next_states_prob = {}
  for action in ACTIONS:
    if action == A:
      this_prob = 0.85
    else:
      this_prob = 0.05
    this_next_state = next_state(S, action)
    if this_next_state not in next_states_prob:
      next_states_prob[this_next_state] = 0
    next_states_prob[this_next_state] = next_states_prob[this_next_state] + this_prob
  
  if T in next_states_prob:
    return next_states_prob[T]
  else:
    return 0

"""### Extra Credit: `transition_model_portal(T,S,A)` [5 points]

You will be implementing the same transition model as above; however, there are portals in the grid world now. The portals are as such:

1. Going Right from (3,4) takes you to (6,3)
2. Going Up from (0,1) takes you to (5,5)
3. Going Left from (3,8) takes you to (0,0)

Once again, you will be using the following statements to calculate the probability:

$T$ is the correct state when you take action $A$ from $S$; $P(T|S,A) = 0.85$ \\
$T$ is a possible state when you take an action other than $A$ from $S$; $P(T|S,A) = 0.05$ \\
Otherwise; $P(T|S,A) = 0$

**NOTE**: If the robot is at the edge of the grid world, and tries to perform an action that will take it out of the grid world, it will stay in the same state (location), unless a portal dictates otherwise.
"""

def transition_model_portal(T, S, A):
  """Give value of CPT for transition model P(T|S,A) with portals.
    arguments:
        T (int,int): a tuple (i,j), representing the next position in the grid
        S (int,int): a tuple (i,j), representing the position in the grid.
        A (str): a string representing the action to be taken
    returns:
        P(T|S,A) (float).
  """
  def next_state(S, A):
    transition_offset = {
      'Left': (0, -1),
      'Right': (0, 1),
      'Up': (-1, 0),
      'Down': (1, 0)
    }
    portals = {
        ((3, 4), 'Right'): (6, 3),
        ((0, 1), 'Up'): (5, 5),
        ((3, 8), 'Left'): (0, 0)
    }
    if (S, A) in portals:
      return portals[(S, A)]
    temp = (S[0] + transition_offset[A][0], S[1] + transition_offset[A][1])
    if temp in STATES:
      return temp
    else:
      return S
   
  next_states_prob = {}
  for action in ACTIONS:
    if action == A:
      this_prob = 0.85
    else:
      this_prob = 0.05
    this_next_state = next_state(S, action)
    if this_next_state not in next_states_prob:
      next_states_prob[this_next_state] = 0
    next_states_prob[this_next_state] = next_states_prob[this_next_state] + this_prob
  
  if T in next_states_prob:
    return next_states_prob[T]
  else:
    return 0

"""## Sampling

### 1. `calculate_cdf(pmf, ordered_outcomes)` [5 points]

This function will take in a probability mass function (pmf) and all the possible outcomes the pmf can take. For example, if we consider the cdf for actions, the variable `ordered_outcomes` will be `[Left, Right, Up, Down]`. You will be returning a function `cdf(outcome)`, which returns the cumulative probability of that outcome. For example, in the table below, that would be $F(a_i)$. **HINT**: Numpy will be useful here!

$a_{i}$ | $P$$($$A$$=$$a_{i}$$)$ | $F$$($$a_{i}$$)$
--- | --- | ---
Left | 0.2 | 0.2
Right | 0.6 | 0.8
Up | 0.1 | 0.9
Down | 0.1 | 1.0
"""

# Create the function cdf
def calculate_cdf(pmf, ordered_outcomes):
  """Calculate cumulative distribution function.
  arguments:
      pmf (function): probability mass function.
      ordered_outcomes (list): all outcomes in an ordered list.
  returns:
      a function that represents the CDF
  note:
      The ordered_outcomes defines how the CDF behaves.
  """
  
  def cdf(outcome):
    pmf_result = np.array([pmf(outcome) for outcome in ordered_outcomes])
    cdf_list = np.cumsum(pmf_result)
    look_up = dict(zip(ordered_outcomes, cdf_list))
    return look_up[outcome]
    

  return cdf

"""### 2. `sample_from_pmf(pmf, ordered_outcomes)` [6 points]
 
This function will take in a pmf and all the possible outcomes the pmf can take, and the goal is to sample from this pmf, using inverse transform sampling. You will be returning either an action or state, depending on the pmf and ordered outcomes.

**HINT**: Numpy will be useful here! 

Please use `generate_random_float` to get a random float between 0 and 1.
"""

# Fill in the body of the following function to sample from an arbitrary PMF.
def sample_from_pmf(pmf, ordered_outcomes):
  """Sample from a PMF
    arguments:
      pmf (function): a probability mass function
      ordered_outcomes (list): ordered list of all outcomes
    returns:
      a state or an action, depending on the pmf and outcomes
  """
  cdf = calculate_cdf(pmf, ordered_outcomes)
  rand_float = generate_random_float()
  for outcome in ordered_outcomes:
    if cdf(outcome) > rand_float:
      return outcome

"""### 3. `sample_from_sensor_model(state)` [3 points]

This function is similar to `sample_from_pmf`. The only difference is that this is specific to the sensor model, and hence, it has the additional parameter `state`, referring to the $S$ in $P(O|S)$. So, in this case, your pmf will be `sensor_model`. You will be returning an observation, which is a value of the sensor. 

**HINT**: Numpy will be useful here!

Please use `generate_random_float` to get a random float between 0 and 1.
"""

def sample_from_sensor_model(state):
  """Sample from sensor model
  arguments: 
      state (int, int): a pair (i,j), representing the position in the grid.
  returns:
      observation (int): the sensor reading that has been sampled
  """
  def pmf(k):
    return sensor_model(k, state)

  k_list = list(np.arange(STATES[0][0], STATES[-1][0] + 1))
  
  return sample_from_pmf(pmf, k_list)

"""### 4. `sample_from_transition_model(state, action)` [3 points]
This function is also similar to `sample_from_pmf`. The difference is that this is specific to the transition model, and hence, it has additional parameters `state` and `action`, referring to the $S$ and $A$ in $P(T|S,A)$. So, in this case, your pmf will be `transition_model`. You will be returning a state $T$.

**HINT**: Numpy will be useful here!

Please use `generate_random_float` to get a random float between 0 and 1.
"""

def sample_from_transition_model(state, action):
  """Sample from transition model
  arguments:
      state (int, int): a pair (i,j), representing the position in the grid.
      action (str): the action taken
  returns:
      next_state (int, int): a pair (i,j), representing the robot's next position in the grid
  """
  def pmf(T):
    return transition_model(T, state, action)
   
  def next_state(S, A):
    transition_offset = {
      'Left': (0, -1),
      'Right': (0, 1),
      'Up': (-1, 0),
      'Down': (1, 0)
    }
    temp = (S[0] + transition_offset[A][0], S[1] + transition_offset[A][1])
    if temp in STATES:
      return temp
    else:
      return S
  
  return sample_from_pmf(pmf, list(set([next_state(state, some_action) for some_action in ACTIONS])))

"""## Inference

![](https://i.ibb.co/QmdLWvh/Screen-Shot-2020-01-21-at-7-26-56-AM.png)

### 1. `sample_from_dbn()` [9 points]

In this function, you will be sampling from a dynamic Bayes net. You will need to perform the sampling for the Bayes net shown above, which has three observations, three states and two actions. You will be returning all the sampled states, observations and actions in a dictionary `samples`. The sampling functions you have written earlier will be helpful in this. Make sure you go through the slides, as this will be helpful in creating your algorithm.
"""

def sample_from_dbn():
  """Sample from a dynamic bayes network
  returns:
      samples (dict): A dictionary with the states, observations and actions sampled
  """
  samples = {'states': [], 'observations': [], 'actions': []}
  # Your code below

  s1 = sample_from_pmf(state_prior, STATES)
  o1 = sample_from_sensor_model(s1)
  a1 = sample_from_pmf(action_prior, ACTIONS)
  s2 = sample_from_transition_model(s1, a1)
  o2 = sample_from_sensor_model(s2)
  a2 = sample_from_pmf(action_prior, ACTIONS)
  s3 = sample_from_transition_model(s2, a2)
  o3 = sample_from_sensor_model(s3)
  samples['states'].append(s1)
  samples['states'].append(s2)
  samples['states'].append(s3)
  samples['observations'].append(o1)
  samples['observations'].append(o2)
  samples['observations'].append(o3)
  samples['actions'].append(a1)
  samples['actions'].append(a2)

  return samples

"""### 2. `maximum_probable_explanation(actions, observations)` [12 points]
In this function, you will implement MPE, which infers the most probable values of the states, given the actions and observations. You will be implementing MPE for the Bayes net above. You are asked to maximize the posterior probability
$$P(S_1, S_2, S_3|O_1=o_1, A_1=a_1, O_2=o_2, A_2=a_2, O_3=o_3)$$
To do this, you will implement a highly specialized version of the max-product algorithm. You will be returning the three states $S_1, S_2, S_3$ in a list.

**Note:** there are three helper functions below. They are partly implemented and is there to aid you through the algorithm, but our way of implementation is not required as long as the main function `maximum_probable_explanation` works correctly.
"""

def create_lookup_table_for_s1(o1, a1):
  """Create a lookup table for state 1
    arguments:
      o1 (int): observation 1
      a1 (str): action 1
    returns:
      g1 (dict): lookup table for s1
      v1 (dict): value table for s1
  """
  def product1(s1, s2):
    """Inner function to calculate the factor value of state 1 and 2
      arguments:
        s1 (int, int): state 1
        s2 (int, int): state 2
      returns:
        value of the factor (float)
    """
    return state_prior(s1) * sensor_model(o1, s1) * transition_model(s2, s1, a1)
    
  # Your code below for create_lookup_table_for_s1
  g1 = {}
  v1 = {}
  for s2 in STATES:
    prd_list = []
    s_list = []
    for s1 in STATES:
      prd = product1(s1, s2)
      s_list.append(s1)
      prd_list.append(prd)
    v1[s2] = max(prd_list)
    g1[s2] = s_list[prd_list.index(v1[s2])]

  # Your code above
  return g1, v1

def create_lookup_table_for_s2(o2, a2, v1):
  """Create a lookup table for state 2
    arguments:
      o2 (int): observation 2
      a2 (str): action 2
      v1 (dict): value table for state 1
    returns:
      g2 (dict): lookup table for s2
      v2 (dict): value table for s2
  """
  def product2(s2, s3):
    """Inner function to calculate the factor value of state 2 and 3
      arguments:
        s2 (int, int): state 2
        s3 (int, int): state 3
      returns:
        value of the factor (float)
    """
    return sensor_model(o2, s2) * transition_model(s3, s2, a2)

  g2 = {}
  v2 = {}
  # Your code below for create_lookup_table_for_s2
  for s3 in STATES:
    prd_list = []
    s_list = []
    for s2 in STATES:
      prd = v1[s2] * product2(s2, s3)
      s_list.append(s2)
      prd_list.append(prd)
    v2[s3] = max(prd_list)
    g2[s3] = s_list[prd_list.index(v2[s3])]

  # Your code above
  return g2, v2


def create_lookup_table_for_s3(o3, v2):
  """Create a lookup table for state 3
    arguments:
      o3 (int): observation 3
      v2 (dict): value table for s2
    returns:
      g3 (state): lookup table for s3
      v3 (float): value table for s3
  """
  def product3(s3):
    """Inner function to calculate the factor value of state 2 and 3
      arguments:
        s3 (int, int): state 3
      returns:
        value of the factor (float)
    """
    return sensor_model(o3, s3)

  g3 = None
  v3 = None
  # Your code below for create_lookup_table_for_s3
  prd_list = []
  s_list = []
  for s3 in STATES:
    prd = v2[s3] * product3(s3)
    s_list.append(s3)
    prd_list.append(prd)
  v3 = max(prd_list)
  g3 = s_list[prd_list.index(v3)]
  # Your code above
  return g3, v3

def maximum_probable_explanation(actions, observations):
  """Calculate the most probable states given the actions and observations
      arguments:
        actions ([str]): Array of all actions taken
        observations ([int]): Sensor measurements
      returns:
        list of most probable states ([(int, int)])
  """
  # Known values
  o1, o2, o3 = observations
  a1, a2 = actions

  # eliminate S1 -> lookup table S1=g1(S2)
  g1, v1 = create_lookup_table_for_s1(o1, a1)

  # eliminate S2 -> lookup table S2=g2(S3)
  g2, v2 = create_lookup_table_for_s2(o2, a2, v1)

  # eliminate S3 -> lookup table S3=g3()
  g3, v3 = create_lookup_table_for_s3(o3, v2)

  # back-substitute (use the lookup tables to find the most probable states)
  # Your code below

  return[g1[g2[g3]], g2[g3], g3]

"""### EXTRA CREDIT: `general_maximum_probable_explanation(actions, observations)` [5 points]

You can receive extra credit if your implementation of MPE is for a Bayes Net having the same structure as above, but having any number of states. So, if you're given $N$ observations and $N - 1$ actions, you will be returning $N$ states.
"""

def general_maximum_probable_explanation(actions, observations):
  """Calculate the most probable states given the actions and observations
      arguments:
        actions ([str]): Array of all actions taken
        observations ([int]): Sensor measurements
      returns:
        list of most probable states.
  """
  if len(actions) == 0:
    # apply simple bayes rule here
    likelihood = {}
    max_state = None
    max_val = None
    for state in STATES:
      likelihood[state] = sensor_model(observations[0], state) * state_prior(state)
      if max_val is None or likelihood[state] > max_val:
        max_val = likelihood[state]
        max_state = state
    return [state]
    
  actions = actions.copy()
  observations = observations.copy()
  lookup_tables = [];
  start = True
  while True:
    if start:
      o1 = observations.pop(0)
      a1 = actions.pop(0)
      g1, curr_v = create_lookup_table_for_s1(o1, a1)
      lookup_tables.append(g1)
      start = False
    else:
      if len(actions) != 0:
        curr_o = observations.pop(0)
        curr_a = actions.pop(0)
        curr_g, curr_v = create_lookup_table_for_s2(curr_o, curr_a, curr_v)
        lookup_tables.append(curr_g)
      else:
        curr_o = observations.pop(0)
        curr_g, curr_v = create_lookup_table_for_s3(curr_o, curr_v)
        lookup_tables.append(curr_g)
        break
  
  states = []
  curr_state = lookup_tables.pop()
  states.append(curr_state)
  while len(lookup_tables) != 0:
    curr_table = lookup_tables.pop()
    curr_state = curr_table[curr_state]
    states.insert(0, curr_state)
    
  return states

"""## Testing

We have provided some unit tests for you to run to make sure that your code is working properly. However, these are not the exact same unit tests we will be running to make sure your code works properly, so your performance on these unit tests doesn't equate to your final score. Make sure you add a screenshot of the results of the tests in the appropriate slide in the reflection.

**NOTE**: Please don't hard code solutions to pass these unit tests, as these won't be the exact same unit tests we will be running.
"""

import unittest
from collections import Counter

class TestFunctions(unittest.TestCase):
    def test_functions_exist(self) -> None:
        """
        Checks if all the necessary functions exist.
        Autograder shall give an error at appropriate places if they don't.
        This will inform the user if some missing function caused the error.
        """
        assert 'action_prior' in globals(), "action_prior is missing"
        assert 'state_prior' in globals(), "state_prior is missing"
        assert 'calculate_cdf' in globals(), "calculate_cdf is missing"
        assert 'sample_from_pmf' in globals(), "sample_from_pmf is missing"
        assert 'sensor_model' in globals(), "sensor_model is missing"
        assert 'transition_model' in globals(), "transition_model is missing"
        assert 'maximum_probable_explanation' in globals(), "maximum_probable_explanation is missing"
        assert 'sample_from_sensor_model' in globals(), "sample_from_sensor_model is missing"
        assert 'sample_from_transition_model' in globals(), "sample_from_transition_model is missing"
        assert 'sample_from_dbn' in globals(), "sample_from_dbn is missing"

    def test_actions_exist(self) -> None:
        """
        Checks if ACTIONS variable exists.
        """
        assert 'ACTIONS' in globals(), "ACTIONS not found"
        self.assertEqual(len(ACTIONS), 4, "ACTIONS does not have four possibilities")

    def test_action_prior(self) -> None:
        """
        Checks if action_prior is working correctly.
        """
        prob_sum = 0
        for action in ACTIONS:
            p = action_prior(action)
            assert p is not None, "None returned by action_prior"
            self.assertGreaterEqual(p, 0, "Probability from action_prior cannot be lesser than 0")
            self.assertLessEqual(p, 1.0, "Probability from action_prior cannot be greater than 1")
            prob_sum += p
        # Takes care of floating-point errors in Python
        # because of how they are stored in memory
        self.assertGreaterEqual(prob_sum, 1.0 - 1e-6, "Sum of probabilities cannot be lesser than 1")
        self.assertLessEqual(prob_sum, 1.0 + 1e-6, "Sum of probabilities cannot be greater than 1")

    def test_states_exist(self) -> None:
        """
        Checks if STATES variable exists.
        """
        assert 'STATES' in globals(), "STATES not found"
        self.assertEqual(len(STATES), 100, "STATES does not have 100 possibilities")

    def test_state_prior(self) -> None:
        """
        Checks if state_prior is working correctly.
        """
        prob_sum = 0
        for state in STATES:
            p = state_prior(state)
            assert p is not None, "None returned by state_prior"
            self.assertGreaterEqual(p, 0, "Probability from state_prior cannot be lesser than 0")
            self.assertLessEqual(p, 1.0, "Probability from state_prior cannot be greater than 1")
            prob_sum += p
        # Takes care of floating-point errors in Python
        # because of how they are stored in memory
        self.assertGreaterEqual(prob_sum, 1.0 - 1e-6, "Sum of probabilities cannot be lesser than 1")
        self.assertLessEqual(prob_sum, 1.0 + 1e-6, "Sum of probabilities cannot be greater than 1")

    def test_calculate_cdf(self) -> None:
        """
        Checks if CDF is working properly.
        """
        cdf = calculate_cdf(action_prior, ACTIONS)
        assert callable(cdf), "calculate_cdf must return a function"
        curr_sum = 0
        assert curr_sum < cdf("Left"), "CDF must go in ascending order for the actions passed in"
        curr_sum = cdf("Left")
        assert 0.2 - 1e-6 < curr_sum < 0.2 + 1e-6, f"Final CDF Value must 0.2, {curr_sum} found"
    
    def pmf_action_sanity_check(self) -> None:
        """
        Checks if pmf returns an action for actions.
        """
        sample_num_10 = [sample_from_pmf(action_prior, ACTIONS)]
        for curr_sample in sample_num_10:
            assert curr_sample in ACTIONS, "PMF returns something other than allowable outputs for actions"

    def test_sensor_model(self) -> None:
        """
        Checks if the sensor model is working properly.
        """
        curr_state = (3, 6)
        for i in range(0, 3):
            assert sensor_model(i, curr_state) == 0.01, \
                "Sensor model does not return 0.01 for incorrect observation."
        assert sensor_model(3, curr_state) == 0.91,\
            "Sensor model does not return 0.01 for incorrect observation."
        for i in range(4, 10):
            assert sensor_model(i, curr_state) == 0.01, \
                "Sensor model does not return 0.01 for incorrect observation."

    def test_transition_model_center_normal(self) -> None:
        """
        Checks if the transition model is working properly if the robot is neither on the edge nor in the corner.
        """
        curr_state = (3, 6)
        curr_action = "Right"
        correct_state = (3, 7)
        incorrect_states = [(3, 5), (2, 6), (4, 6)]
        invalid_states = [(0, 0), (5, 5), (6, 3)]
        assert transition_model(correct_state, curr_state, curr_action) == 0.85,\
            "Correct state does not return 0.85"
        for curr_incorrect_state in incorrect_states:
            assert transition_model(curr_incorrect_state, curr_state, curr_action) == 0.05, \
                "Incorrect state does not return 0.05"
        for curr_invalid_state in invalid_states:
            assert transition_model(curr_invalid_state, curr_state, curr_action) == 0, \
                "Invalid state does not return 0"
      
    def test_transition_model_corner_wall(self) -> None:
      """
      Checks if the transition model is working properly if the robot is on the edge and is moving towards wall.
      """
      curr_state = (0, 0)
      curr_action = "Left"
      correct_state = (0, 0)
      incorrect_states = [(0, 1), (1, 0)]
      invalid_states = [(1, 1), (5, 5), (6, 3)]
      assert transition_model(correct_state, curr_state, curr_action) == 0.90,\
          "Correct state does not return 0.90"
      for curr_incorrect_state in incorrect_states:
          assert transition_model(curr_incorrect_state, curr_state, curr_action) == 0.05, \
              "Incorrect state does not return 0.05"
      for curr_invalid_state in invalid_states:
          assert transition_model(curr_invalid_state, curr_state, curr_action) == 0, \
              "Invalid state does not return 0"

        
    def test_sample_sensor_model(self) -> None:
        """
        Checks if sample_from_sensor_model is working properly.
        """
        correct_state = (1, 1)
        samples = [sample_from_sensor_model(correct_state) for _ in range(10000)]
        hist = np.array([Counter(samples)[obs] for obs in OBSERVATIONS])
        hist = hist / np.sum(hist)  # normalize
        sensor_model_flattened = np.array([0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        np.testing.assert_allclose(hist, sensor_model_flattened, atol=1e-2, rtol=1e-2)

    def test_sample_transition_model(self) -> None:
        """
        Checks if sample_from_transition_model is working properly.
        """
        curr_state = (1, 1)
        action = "Right"
        samples = [sample_from_transition_model(curr_state, action) for _ in range(10000)]
        hist = np.array([Counter(samples)[act] for act in [(1, 2), (0, 1), (2, 1), (1, 0)]])
        hist = hist / np.sum(hist)  # normalize
        transition_model_flattened = np.array([0.85, 0.05, 0.05, 0.05])
        np.testing.assert_allclose(hist, transition_model_flattened, atol=1e-1, rtol=1e-1)

    def test_sample_from_dbn_types(self) -> None:
      """
      Checks if sample_from_dbn returns the correct types.
      """
      samples = sample_from_dbn()
      for state in samples['states']:
        assert state in STATES
      for obs in samples['observations']:
        assert obs in OBSERVATIONS
      for action in samples['actions']:
        assert action in ACTIONS
    
    def test_maximum_probable_explanation(self) -> None:
      """
      Checks if maximum_probable_explanation is working properly.
      """
      actions = ['Down', 'Down']
      observations = [3, 4, 5]
      correct_states = [(3,3), (4,3), (5,3)]
      assert maximum_probable_explanation(actions, observations) == correct_states,\
          "Correct state does not return [(3,3), (3,4), (3,5)]"

def suite():
  functions = [
               'test_functions_exist',
               'test_actions_exist',
               'test_action_prior',
               'test_states_exist',
               'test_state_prior',
               'test_calculate_cdf',
               'pmf_action_sanity_check',
               'test_sensor_model',
               'test_transition_model_center_normal',
               'test_transition_model_corner_wall',
               'test_sample_from_dbn_types',
               'test_maximum_probable_explanation',
               'test_sample_sensor_model',
               'test_sample_transition_model'
               ]
  suite = unittest.TestSuite()
  for func in functions:
    suite.addTest(TestFunctions(func))
  return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

"""### Extra Credit (Portal) Test Case

Run this to check if your `transition_model_portal` passes the given test case. Make sure to add the screenshot of the results in the appropriate slide in the reflection.
"""

class TestPortal(unittest.TestCase):
  def test_transition_model_portal(self) -> None:
      """
      EXTRA CREDIT
      Checks if the transition model is working properly for the portal case.
      """
      curr_state = (3,4)
      curr_action = 'Right'
      correct_state = (6,3)
      incorrect_states = [(3,3), (2,4), (4,4)]
      invalid_states = [(0,0), (1,1), (2,2)]
      assert transition_model_portal(correct_state, curr_state, curr_action) == 0.85,\
          "Correct state does not return 0.85"
      for curr_incorrect_state in incorrect_states:
          assert transition_model_portal(curr_incorrect_state, curr_state, curr_action) == 0.05, \
              "Incorrect state does not return 0.05"
      for curr_invalid_state in invalid_states:
          assert transition_model_portal(curr_invalid_state, curr_state, curr_action) == 0, \
              "Invalid state does not return 0"

if __name__ == '__main__':
  suite = unittest.TestSuite()
  suite.addTest(TestPortal("test_transition_model_portal"))
  runner = unittest.TextTestRunner()
  runner.run(suite)

"""### Extra Credit (General MPE) Test Case

Run this to check if your `general_maximum_probable_explanation` passes the given test case. Make sure to add the screenshot of the results in the appropriate slide in the reflection.
"""

class TestMPE(unittest.TestCase):
  def test_general_maximum_probable_explanation(self) -> None:
      """
      Checks if general_maximum_probable_explanation is working properly.
      """
      actions = ['Down', 'Down', 'Right', 'Up', 'Up', 'Left']
      observations = [3, 4, 5, 5, 4, 3, 3]
      correct_states = [(3,3), (4,3), (5,3), (5,4), (4,4), (3,4), (3,3)]
      assert general_maximum_probable_explanation(actions, observations) == correct_states,\
          "Correct state does not return correctly"

if __name__ == '__main__':
  suite = unittest.TestSuite()
  suite.addTest(TestMPE("test_general_maximum_probable_explanation"))
  runner = unittest.TextTestRunner()
  runner.run(suite)

"""# Reflection [50 Points]

You will be answering the questions in the proj2_report_template.pptx (found in the Files tab on Canvas). Save the PowerPoint file as a PDF and rename it to {FirstName1_LastName1}.pdf.

# Submission Details

### Deliverables
A zip file named {FirstName_LastName.zip} with the following files:
1. Project2.py (Created from converting `.ipynb` to `.py`)
  1. To submit this code, you will need to download this `.ipynb` file as a `.py` file (File -> Download as `.py`).
  2. You will also need to include a screenshot of your unit test results in the reflection PowerPoint (a slide has been marked for that).
2. {FirstName_LastName.pdf}: This is the reflection PDF.

Submit this zip file to Canvas.
"""