import numpy as np
import random

# Settable parameters
exploration_rate = 0.2  # Exploration rate ε

payoff_t = 1  # Payoff t
payoff_r = 0.9  # Payoff r
payoff_p = 0.4  # Payoff p
payoff_s = 0  # Payoff s

batch_size = 100000000  # Batch size K
state_space = [0, 1, 2, 3]  # Number of states S
action_space = [0, 1]  # Number of actions A
inertia = 0.4  # Inertia λ


# Functions
# epsilon-greedy action selection
def choose_action(q_values, state):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(action_space)
    else:
        return np.argmax(q_values[state])


def get_policy(q_values):
    binary_str = ''.join(str(bit) for bit in np.argmax(q_values, axis=1))
    return int(binary_str, 2)


# State transitions
def get_next_state(a1, a2):
    if a1 == 0 and a2 == 0:
        return 0, 0
    elif a1 == 0 and a2 == 1:
        return 1, 2
    elif a1 == 1 and a2 == 0:
        return 2, 1
    else:
        return 3, 3


# Reward function
def get_reward(state):
    if state == 0:
        return payoff_p
    elif state == 1:
        return payoff_t
    elif state == 2:
        return payoff_s
    else:
        return payoff_r


# Initialize Q-tables and counts
# Agent 1
q_env_1 = np.random.rand(len(state_space), len(action_space))
q_act_1 = q_env_1.copy()
counts_1 = np.zeros((len(state_space), len(action_space)))
print("Initial policy for agent 1: ", get_policy(q_act_1))

# Agent 2
q_env_2 = np.random.rand(len(state_space), len(action_space))
q_act_2 = q_env_2.copy()
counts_2 = np.zeros((len(state_space), len(action_space)))
print("Initial policy for agent 2: ", get_policy(q_act_2))

# Initialize state
state_1 = random.choice(state_space)
if state_1 == 1 or state_1 == 2:
    state_2 = 3 - state_1
else:
    state_2 = state_1

while True:
    # loop over the batch
    for t in range(batch_size):
        if t % (batch_size/10) == 0:
            print(get_policy(q_act_1), " ", get_policy(q_act_2), " ", t)
        action_1 = choose_action(q_act_1, state_1)
        action_2 = choose_action(q_act_2, state_2)
        next_state_1, next_state_2 = get_next_state(action_1, action_2)

        reward_1 = get_reward(next_state_1)
        counts_1[state_1, action_1] += 1
        alpha_1 = 1 / (counts_1[state_1, action_1] + 1)
        vg_1 = np.sum(q_env_1) / (len(state_space) * len(action_space))
        q_env_1[state_1, action_1] = (1 - alpha_1) * q_env_1[state_1, action_1] + alpha_1 * (reward_1 + 0.5 * exploration_rate * np.min(q_env_1[next_state_1]) + (1 - 0.5 * exploration_rate) * np.max(q_env_1[next_state_1]) - vg_1)

        reward_2 = get_reward(next_state_2)
        counts_2[state_2, action_2] += 1
        alpha_2 = 1 / (counts_2[state_2, action_2] + 1)
        vg_2 = np.sum(q_env_2) / (len(state_space) * len(action_space))
        q_env_2[state_2, action_2] = (1 - alpha_2) * q_env_2[state_2, action_2] + alpha_2 * (reward_2 + 0.5 * exploration_rate * np.min(q_env_2[next_state_2]) + (1 - 0.5 * exploration_rate) * np.max(q_env_2[next_state_2]) - vg_2)

        state_1, state_2 = next_state_1, next_state_2

    # Update policy if random variable greater than inertia
    if random.uniform(0, 1) > inertia:
        q_act_1, q_act_2 = q_env_1.copy(), q_env_2.copy()
    q_env_1, q_env_2 = q_act_1.copy(), q_act_2.copy()
    print("New policies: ", get_policy(q_act_1), " ", get_policy(q_act_2))







