from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def plot_values(V):
	# reshape value function
	V_sq = np.reshape(V, (4,4))

	# plot the state-value function
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111)
	im = ax.imshow(V_sq, cmap='cool')
	for (j,i),label in np.ndenumerate(V_sq):
	    ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
	plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
	plt.title('State-Value Function')
	plt.show()

env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    #render_mode="human",
)

## Testing variables

# print the state space and action space
print(env.observation_space)
print(env.action_space)

# print the total number of states and actions
nS = env.observation_space.n
nA = env.action_space.n
print(nS)
print(nA)

env.P[1][0]


## Iterative Policy Evaluation
#
# V = Array of [N] N number of states, with the estimated value in the actual policy
# delta = max estimated value
# Vs = Estimated value for the actual state
# gamma = reward corrector for next state estimated value
# theta = very small positive number that is used to decide if the estimate has sufficiently converged to the true value function
#

def policy_evaluation(nS, env, policy, gamma=1, theta=1e-8):
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

random_policy = np.ones([nS, nA]) / nA

# evaluate the policy 
V = policy_evaluation(nS, env, random_policy)

plot_values(V)

## Obtaining qπ
#
# q = Array of [N] N number of actions, with the estimated value of each action in the state s
# gamma = discount rate

def q_from_v(nA, env, V, s, gamma=1):
    q = np.zeros(nA)
    for a in range(nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

Q = np.zeros([nS, nA])
for s in range(nS):
    Q[s] = q_from_v(nA, env, V, s)
print("Action-Value Function:")
print(Q)

## Policy improvement
#
# V = Array of [N] N number of states, with the estimated value in the actual policy
# gamma = discount rate
# policy = Array of [N States][N Actions] with the probabilities of taking each action on each state
#

def policy_improvement(nA, nS, env, V, gamma=1):
    policy = np.zeros([nS, nA]) / nA
    for s in range(nS):
        q = q_from_v(nA, env, V, s, gamma)
        
        # OPTION 1: construct a deterministic policy 
        # policy[s][np.argmax(q)] = 1
        
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        best_a = np.argwhere(q==np.max(q)).flatten() # Takes the action with biggest value (if there are equivalents, returns array with the actions)
        
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy

#policy = policy_improvement(nA, nS, env, V)
#print(policy)

