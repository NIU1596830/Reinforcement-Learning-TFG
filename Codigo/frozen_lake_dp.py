from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import copy
import random

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import time # Time calculation
from memory_profiler import profile # Memory calculation
from codecarbon import EmissionsTracker # Energy report


sns.set_theme()

############################ PARAMETERS #######################################

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    gamma: float  # Discounting rate
    theta: float  # Very small positive number that is used to decide if the estimate has sufficiently converged to the true value function
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
    total_episodes=10,
    gamma=1,
    theta=1e-8,
    seed=123,
    is_slippery=True,
    n_runs=5,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./../Media/img/dynamic_programming/"),
    max_steps = 250,
    map_sizes = [4,7,9,11],
    calculate_energy = False,
    plot_results = True,
)
params

###################### Dynamic programming functions #########################

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

## Policy iteration
#

def policy_iteration(nA, nS, env, gamma=1, theta=1e-8):
    policy = np.ones([nS, nA]) / nA
    while True:
        V = policy_evaluation(nS, env, policy, gamma, theta)
        new_policy = policy_improvement(nA, nS, env, V)
        
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;
        
        policy = copy.copy(new_policy)
    return policy, V

## Value iteration
#

def value_iteration(nA, nS, env, gamma=1, theta=1e-8):
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            V[s] = max(q_from_v(nA, env, V, s, gamma))
            delta = max(delta,abs(V[s]-v))
        if delta < theta:
            break
    policy = policy_improvement(nA, nS, env, V, gamma)
    return policy, V

def index_max_random(row):
    max_value = np.max(row)
    max_row = np.where(row == max_value)[0]
    return random.choice(max_row)

@profile
def run_env(): # This will be our main function to run our environment until the maximum number of episodes
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                # Select action by policy
                action = index_max_random(policy_vi[state])
                
                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

    return rewards, steps, episodes, all_states, all_actions

############################  VISUALIZATION  ##################################

def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def dp_get_decision(policy_vi):
    """ Get the best direction for each row, considering
        -1: " ",
        0: "←", 0.1: "←|↓", 0.2: "←|→", 0.3:"←|↑",
        1: "↓", 1.2: "↓|→", 1.3: "↓|↑",
        2: "→", 2.3: "→|↑",
        3: "↑"
    """
    size = policy_vi.shape[0]
    dp_decision = []
    heatmap = []
    
    for row in policy_vi:
        if max(row) == 0.25: # Equivalent decision = hole or finish
            dp_decision.append(-1)
            heatmap.append(0)
        elif max(row) == 1:
            max_value = max(row)
            indexs = [index for index in range(len(row)) if row[index] == max_value]
            dp_decision.append(indexs[0])
            heatmap.append(1)
        else:
            indexs = np.array(row.argmax)
            max_value = max(row)
            indexs = [index for index in range(len(row)) if row[index] == max_value]
            print("AQUIIIII")
            print(row)
            print("---")
            print(indexs)
            div = 1
            value = 0
            counter = 0
            for next_index in indexs:
                value = value + (next_index / div)
                div = div*10
                counter += 1
            dp_decision.append(value)
            heatmap.append(1/counter)
            
    print(policy_vi)
    print("----------")
    print(dp_decision)
    
    return dp_decision, heatmap

def policy_directions_map(policy_vi, map_size):
    """Get the best learned action & map it to arrows."""
    dp_decision, heatmap = dp_get_decision(policy_vi)
    directions = {-1: " ",
                  0: "←", 0.1: "←↓", 0.2: "←→", 0.3:"←↑",
                  1: "↓", 1.2: "↓→", 1.3: "↓↑",
                  2: "→", 2.3: "→↑",
                  3: "↑",
                  1.23: "↓→↑"
                  }
    # TODO: Change to more optimal function with float.is_integer()
    # Change policy array to directions array
    for idx, val in enumerate(dp_decision):
        dp_decision[idx] = directions[val]

    return dp_decision, heatmap

def plot_policy_map(policy_vi, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    dp_decision, heatmap = policy_directions_map(policy_vi, map_size)
    heatmap = np.array(heatmap)
    heatmap = heatmap.reshape((map_size,map_size))
    print(heatmap)
    dp_decision = np.array(dp_decision)
    dp_decision = dp_decision.reshape((map_size,map_size))
    print(dp_decision)
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
    img_title = f"frozenlake_dp_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def winrate(rewards,n_episodes,n_runs):
    # Crear un array para almacenar los conteos de 0 y 1
    episodes_per_it = int(n_episodes/10)
    print(episodes_per_it)
    conteos = np.zeros((10, 2), dtype=int)
    
    # Iterar sobre cada columna de la matriz
    for i in range(n_runs):
        columna = rewards[:, i]
    
        for j in range(10):
            new_columna=columna[(episodes_per_it)*j:(episodes_per_it)*(j+1)]
            # Contar los elementos iguales a 0 y 1 en la columna
            #conteos[(episodes_per_it)*j:(episodes_per_it)*(j+1), 0] = np.count_nonzero(new_columna == 0)
            #conteos[(episodes_per_it)*j:(episodes_per_it)*(j+1), 1] = np.count_nonzero(new_columna == 1)
            conteos[:, 0] = np.count_nonzero(new_columna == 0)
            conteos[:, 1] = np.count_nonzero(new_columna == 1)
            # Imprimir los conteos
            print(f"Map {map_size} Run number {i}, Percentile number {j*10}%: Loses = {conteos[j, 0]}, Wins = {conteos[j, 1]}, Winrate = {int(conteos[j,1]/(episodes_per_it)*100)}")

###############################################################################

# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)

map_sizes = params.map_sizes
res_all = pd.DataFrame()
st_all = pd.DataFrame()

if (params.calculate_energy):
    tracker = EmissionsTracker(project_name="Value Iteration")
    tracker.start()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
        max_episode_steps = params.max_steps,
    )
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    
    # Start timer
    start = time.time()
    
    # Run environment
    policy_vi, V_vi = value_iteration(nA, nS, env, params.gamma, params.theta)
        
    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(policy_vi,"\n")
        
    print(f"Map size: {map_size}x{map_size}")
        
    rewards, steps, episodes, all_states, all_actions = run_env()
    
    
    if(params.plot_results):
        # Save the results in dataframes
        res, st = postprocess(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
    
        plot_states_actions_distribution(
            states=all_states, actions=all_actions, map_size=map_size
        )  # Sanity check
        
        plot_policy_map(policy_vi, env, map_size)
        winrate(rewards,params.total_episodes,params.n_runs)
    env.close()

# Stop timer
end = time.time()

# Calculate time
time = end - start

print(f"Time of execution map {map_size}: {time} seconds")

if (params.calculate_energy):
    tracker.stop()