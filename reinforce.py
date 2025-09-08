

import logging
import math
from matplotlib import pyplot as plt
import numpy as np
from quasim.gates import (
    H,
    S,
    T,
    CX,
    IGate,
    Gate,
    CGate
)
from random import seed
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Union

from utils.circuit import create_random_circuit, create_random_states, update_state, decomplexify_vector
from utils.metrics import min_entanglement_entropy
from utils.rl import sample_epsilon_greedy, sample, sample_best


MAX_QUBITS = 2
GATE_COUNT = 4  # Clifford gate set


def measure_distance_to_target(state: np.ndarray) -> float:
    target_state = np.array([1, 0, 0, 0])

    total_distance = 0
    for xi, yi in zip(state, target_state):
        total_distance += abs(xi - yi)

    # return 1 - (total_distance / 2) # maximum distance is 2
    return total_distance


def compute_reward(state: np.ndarray, new_state: np.ndarray, punishment_term: float = 0.1) -> float:
    distance_1 = min_entanglement_entropy(state)

    if distance_1 < 0.01 and new_state is None:
        return 5
    else:
        return - punishment_term


    if new_state is None: # case: previous step output was no new gate
        if distance_1 < 0.01:
            return 2
        else:
            return -distance_1

    distance_2 = min_entanglement_entropy(new_state)
    return - (distance_2 - distance_1) - punishment_term


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        # input layer
        # times 2 because state is split into real and imaginary
        self.hidden1 = nn.Linear(2 * 2 ** MAX_QUBITS, 128)

        # actor's layer
        self.gate_type_pred = nn.Linear(128, GATE_COUNT + 1)
        self.control_qubit_pred = nn.Linear(128, MAX_QUBITS)
        self.target_qubit_pred = nn.Linear(128, MAX_QUBITS)

    def forward(self, state):
        state = torch.tensor(decomplexify_vector(state), dtype=torch.float32)

        x = F.relu(self.hidden1(state))

        gate_type_probs = F.softmax(self.gate_type_pred(x), dim=-1)
        control_qubit_probs = F.softmax(self.control_qubit_pred(x), dim=-1)
        target_qubit_probs = F.softmax(self.target_qubit_pred(x), dim=-1)

        return gate_type_probs, control_qubit_probs, target_qubit_probs


def get_gate(gate_type_idx, control_qubit_idx, target_qubit_idx) -> Union[IGate, None]:
    # Case: stop!
    if gate_type_idx == 0:
        return None

    if gate_type_idx == 1:
        return H(target_qubit_idx)
    elif gate_type_idx == 2:
        return S(target_qubit_idx)
    elif gate_type_idx == 3:
        return T(target_qubit_idx)
    elif gate_type_idx == 4:

        # TODO: Find better solution
        if control_qubit_idx == target_qubit_idx:
            control_qubit_idx = 0
            target_qubit_idx = 1

        # control qubit idx should be 0 if not a controlled gate.
        return CX(control_qubit_idx, target_qubit_idx)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    # ACTOR WITHOUT CRITIC

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    seed(1)
    torch.manual_seed(0)

    GAMMA = 0.95
    EPISODES = 20000
    EPISODE_LENGTH = 10

    LOG_EVERY = 100
    MOVING_AVERAGE_WINDOW = 5

    actor = Actor()

    optimizer = optim.Adam(actor.parameters(), lr=3e-2)

    episode_rewards = []
    moving_average_episode_rewards = []
    episodes = []

    for episode in range(EPISODES):
        logging.debug(f"Starting episode {episode}")

        actions = []
        rewards = []

        # Start from entangled states to avoid getting stuck in 
        # local optima always proposing none-gate.
        for _ in range(100):
            state = create_random_states(1, gate_count=20, qubit_num=MAX_QUBITS)[0]
            
            if min_entanglement_entropy(state) > 0:
                break 

        logging.debug(f"\tStart state: {state}")

        # generate episode data
        for t in range(EPISODE_LENGTH):

            # select action
            gate_type_probs, control_qubit_probs, target_qubit_probs = actor(
                state)

            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_epsilon_greedy(gate_type_dist, epsilon=0.1)

            control_qubit_dist = Categorical(control_qubit_probs)
            control_qubit = sample_epsilon_greedy(control_qubit_dist)

            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_epsilon_greedy(target_qubit_dist)

            actions.append([
                gate_type_dist.log_prob(gate_type),
                control_qubit_dist.log_prob(control_qubit),
                target_qubit_dist.log_prob(target_qubit)
            ])

            gate = get_gate(gate_type,
                            control_qubit, target_qubit)
            logging.debug(f"\tAdding gate {gate}")

            if gate is not None:
                new_state = update_state(state, gate)

                reward = compute_reward(state, new_state)
                logging.debug(f"\tCurrent reward: {reward}")

                rewards.append(reward)

                state = new_state

            else:

                reward = compute_reward(state, None)
                logging.debug(f"\tCurrent reward: {reward}")

                rewards.append(reward)

                logging.debug(f"\tBreaking episode at t={t}.")
                break

        logging.debug(f"\tEnd state: {state}")

        # discount realized rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + GAMMA * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards)

        if episode % LOG_EVERY == 0:
            episode_reward = float(discounted_rewards.sum())
            logging.info(
                f"Discounted episode {episode} reward: {episode_reward:.2f}")

            episode_rewards.append(episode_reward)
            episodes.append(episode)

            if len(episode_rewards) <= MOVING_AVERAGE_WINDOW:
                moving_average_episode_rewards.append(mean(episode_rewards))
            else:
                moving_average = (sum(
                    moving_average_episode_rewards[-MOVING_AVERAGE_WINDOW + 1:]) + episode_reward) / MOVING_AVERAGE_WINDOW
                moving_average_episode_rewards.append(moving_average)

        # perform backprop
        gate_type_losses = []
        control_qubit_losses = []
        target_qubit_losses = []

        for (gate_type_logprob, control_qubit_logprob, target_qubit_logprob), discounted_reward in zip(actions, discounted_rewards):

            gate_type_losses.append(
                -gate_type_logprob * discounted_reward
            )
            control_qubit_losses.append(
                -control_qubit_logprob * discounted_reward
            )
            target_qubit_losses.append(
                -target_qubit_logprob * discounted_reward
            )

        optimizer.zero_grad()

        torch.stack(gate_type_losses).sum().backward(retain_graph=True)
        torch.stack(control_qubit_losses).sum().backward(retain_graph=True)
        torch.stack(target_qubit_losses).sum().backward(retain_graph=True)

        optimizer.step()

        actions = []
        rewards = []

    plt.plot(episodes, episode_rewards)
    plt.plot(episodes, moving_average_episode_rewards)
    plt.show()

    states = create_random_states(5, gate_count=20, qubit_num=MAX_QUBITS)

    for s, state in enumerate(states):
        print("\n====================")
        print(f"Test state # {s + 1}:")
        print("====================")

        for t in range(EPISODE_LENGTH):
            print(f"\nstate_{t}: {state}")
            print(f"\tDistance: {min_entanglement_entropy(state)}")

            gate_type_probs, control_qubit_probs, target_qubit_probs = actor(
                state)

            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_best(gate_type_dist)

            control_qubit_dist = Categorical(control_qubit_probs)
            control_qubit = sample_best(control_qubit_dist)

            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_best(target_qubit_dist)

            gate = get_gate(gate_type,
                            control_qubit, target_qubit)
            print(f"\tAdding gate {gate}")

            if gate is not None:
                new_state = update_state(state, gate)

                reward = compute_reward(state, new_state)
                rewards.append(reward)

                state = new_state

            else:

                reward = compute_reward(state, None)
                rewards.append(reward)

                break

        print(f"\nFinal state: {state}")
        print(f"\tFinal distance: {min_entanglement_entropy(state)}")
