from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import copy
import random
import operator
from IPython.display import clear_output
from time import sleep
import itertools

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import time # Time calculation
from memory_profiler import profile # Memory calculation
from codecarbon import EmissionsTracker # Energy report

tqdm.monitor_interval = 0

############################ PARAMETERS #######################################
class Params(NamedTuple):
    total_episodes: int  # Total episodes
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved
    max_steps : int # Number of max steps in the environment
    map_sizes : list # List with the dimensions of each map we want to run
    calculate_energy : bool # True for energy report
    plot_results : bool # True for plot results


params = Params(
    total_episodes=2000,
    epsilon=0.01,
    map_size=5,
    seed=123,
    is_slippery=True,
    n_runs=5,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./../Media/img/montecarlo/"),
    max_steps = 250,
    map_sizes = [4,7,9,11],
    calculate_energy = False,
    plot_results = True,
)
params

########################### Montecarlo functions ##############################

## Random policy generator 
#

def create_random_policy(env):
    policy = {}
    for key in range(0, env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[key] = p
    return policy

## Create (state,action) dictionary
#

def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q

## Run environment
#

def run_game(env, policy, display=True):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s

        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
                top_range += prob[1]
                if n < top_range:
                    action = prob[0]
                    break 
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            finished = True
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    return episode

## Winrate
#

def test_policy(policy, env):
    wins = 0
    r = 100
    for i in range(r):
        w = run_game(env, policy, display=False)[-1][-1]
        if w == 1:
                wins += 1
    return wins / r

## Develop montecarlo policy
#

def monte_carlo_e_soft(env, episodes=100, policy=None, epsilon=0.01):
    if not policy:
        policy = create_random_policy(env)  # Create an empty dictionary to store state action values    
    Q = create_state_action_dictionary(env, policy) # Empty dictionary for storing rewards for each state-action pair
    returns = {}
    
    for _ in range(episodes): # Looping through episodes
        G = 0 # Store cumulative reward in G (initialized at 0)
        episode = run_game(env=env, policy=policy, display=True) # Store state, action and value respectively 
        
        # for loop through reversed indices of episode array. 
        # The logic behind it being reversed is that the eventual reward would be at the end. 
        # So we have to go back from the last timestep to the first one propagating result from the future.
        
        for i in reversed(range(0, len(episode))):   
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            G += r_t # Increment total reward by reward on current timestep
            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]: # 
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]   
                    
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes
                
                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)
                
                A_star = max_Q
                
                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))

    return policy

############################  VISUALIZATION  ##################################

def accion_mas_probable(probabilidades):
    acciones_mas_probables = {}

    for posicion, probs in probabilidades.items():
        accion = max(probs, key=probs.get)
        acciones_mas_probables[posicion] = accion

    return acciones_mas_probables

def dp_get_decision(policy_vi):
    """ Get the best direction for each row, considering
        -1: " ",
        0: "←", 0.1: "←|↓", 0.2: "←|→", 0.3:"←|↑",
        1: "↓", 1.2: "↓|→", 1.3: "↓|↑",
        2: "→", 2.3: "→|↑",
        3: "↑"
    """
    size = len(policy_vi)
    dp_decision = []
    heatmap = []

    for row in policy_vi.values():
        if max(row.values()) == 0.25:  # Equivalent decision = hole or finish
            dp_decision.append(-1)
            heatmap.append(0)
        elif max(row.values()) == 1:
            max_value = max(row.values())
            indexs = [index for index in row if row[index] == max_value]
            dp_decision.append(indexs[0])
            heatmap.append(1)
        else:
            indexs = [index for index in row if row[index] == max(row.values())]
            div = 1
            value = 0
            counter = 0
            for next_index in indexs:
                value = value + (next_index / div)
                div = div * 10
                counter += 1
            dp_decision.append(value)
            heatmap.append(1 / counter)

    return dp_decision, heatmap


def policy_directions_map(policy_vi, map_size):
    """Get the best learned action & map it to arrows."""
    dp_decision, heatmap = dp_get_decision(policy_vi)
    directions = {
        -1: " ",
        0: "←", 0.1: "←↓", 0.2: "←→", 0.3: "←↑",
        1: "↓", 1.2: "↓→", 1.3: "↓↑",
        2: "→", 2.3: "→↑",
        3: "↑",
        1.23: "↓→↑"
    }
    # TODO: Change to more optimal function with float.is_integer()
    # Change policy dictionary to directions array
    for idx, val in enumerate(dp_decision):
        dp_decision[idx] = directions[val]

    return dp_decision, heatmap


def plot_policy_map(policy_vi, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    dp_decision, heatmap = policy_directions_map(policy_vi, map_size)
    heatmap = np.array(heatmap)
    heatmap = heatmap.reshape((map_size, map_size))
    dp_decision = np.array(dp_decision)
    dp_decision = dp_decision.reshape((map_size, map_size))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    # Plot the last frame
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        heatmap,
        annot=dp_decision,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={'size': 15},
    ).set(title="Optimal Policy\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_montecarlo_{map_size}x{map_size}.png"
    fig.savefig(img_title, bbox_inches="tight")
    plt.show()

###############################################################################

map_size = 11

# Start timer
start = time.time()

if (params.calculate_energy): # Calculate energy
    tracker = EmissionsTracker(project_name="Montecarlo")
    tracker.start()

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=map_size, p=params.proba_frozen, seed=params.seed
    ),
    max_episode_steps = params.max_steps,
)
policy = monte_carlo_e_soft(env, episodes=70000)

print(test_policy(policy, env))

if (params.plot_results):
    plot_policy_map(policy, env, map_size)

if(params.calculate_energy):
    tracker.stop()

# Stop timer
end = time.time()
    
# Calculate time
time = end - start

print(f"Time of execution: {time} seconds")