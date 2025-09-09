

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
    CGate,
    Phase
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
GATE_TYPE_COUNT = 4  # Clifford gate set


def one_hot_encode(dims: int, target_dim: int) -> np.ndarray:
    encoding = [0] * dims
    encoding[target_dim] = 1
    return np.array(encoding)


def measure_distance_to_target(state: np.ndarray) -> float:
    target_state = np.array([1, 0, 0, 0])

    total_distance = 0
    for xi, yi in zip(state, target_state):
        total_distance += abs(xi - yi)

    # return 1 - (total_distance / 2) # maximum distance is 2
    return total_distance


def compute_reward(state: np.ndarray, punishment_term: float = 2) -> float:
    distance = min_entanglement_entropy(state)

    if distance == 0:
        return 10
    else:
        return - punishment_term


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()
        # times 2 because state is split into real and imaginary
        self.input_layer = nn.Linear(2 * 2 ** MAX_QUBITS, 128)

        self.gate_type_hidden = nn.Linear(128, 128)
        self.target_qubit_hidden = nn.Linear(128, 128)
        self.control_qubit_hidden = nn.Linear(128, 128)

        # Avoid none output for now.
        self.gate_type_output = nn.Linear(128, GATE_TYPE_COUNT)
        self.target_qubit_output = nn.Linear(128, MAX_QUBITS)
        self.control_qubit_output = nn.Linear(128, MAX_QUBITS)

    def forward(self, state: np.ndarray):
        state = torch.tensor(decomplexify_vector(state), dtype=torch.float32)

        x = F.relu(self.input_layer(state))

        gate_type_x = F.relu(self.gate_type_hidden(x))
        target_qubit_x = F.relu(self.target_qubit_hidden(x))
        control_qubit_x = F.relu(self.control_qubit_hidden(x))

        gate_type_probs = F.softmax(self.gate_type_output(gate_type_x), dim=-1)
        target_qubit_probs = F.softmax(
            self.target_qubit_output(target_qubit_x), dim=-1)
        control_qubit_probs = F.softmax(
            self.control_qubit_output(control_qubit_x), dim=-1)

        return gate_type_probs, target_qubit_probs, control_qubit_probs


def get_gate(gate_type_idx, target_qubit_idx, control_qubit_idx=None) -> Union[IGate, None]:
    if gate_type_idx == 0:
        return H(target_qubit_idx)
    elif gate_type_idx == 1:
        return S(target_qubit_idx)
    elif gate_type_idx == 2:
        return T(target_qubit_idx)
    elif gate_type_idx == 3:

        # Workaround to avoid same value target and control qubits.
        # Identity gate was not available in current version of
        # quasim.
        if control_qubit_idx == target_qubit_idx:
            return Phase(target_qubit_idx, 0)

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

    agent = Agent()

    optimizer = optim.Adam(agent.parameters(), lr=3e-2)

    episode_rewards = []
    moving_average_episode_rewards = []
    episodes = []

    EPSILON = 1

    GATE_LOG = {}

    for episode in range(EPISODES):
        logging.debug(f"Starting episode {episode}")

        actions = []
        rewards = []

        # Start from entangled states to avoid getting stuck in
        # local optima always proposing none-gate.
        for _ in range(100):
            state = create_random_states(
                1, gate_count=20, qubit_num=MAX_QUBITS)[0]

            if min_entanglement_entropy(state) > 0:
                break

        logging.debug(f"\tStart state: {state}")

        # half epsilon after every 2000 episodes
        if episode > 0 and (episode + 1) % 1000 == 0:
            print(f"Epsilon: {EPSILON}")
            EPSILON = EPSILON * 0.8

            from pprint import pprint
            pprint(GATE_LOG)
            GATE_LOG = {}

        # generate episode data
        for t in range(EPISODE_LENGTH):

            if min_entanglement_entropy(state) == 0:
                logging.debug(f"\tBreaking episode at t={t}.")
                break

            gate_type_probs, target_qubit_probs, control_qubit_probs = agent(
                state)

            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_epsilon_greedy(gate_type_dist, epsilon=EPSILON)

            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_epsilon_greedy(
                target_qubit_dist, epsilon=EPSILON)

            control_qubit_dist = Categorical(control_qubit_probs)
            control_qubit = sample_epsilon_greedy(
                control_qubit_dist, epsilon=EPSILON)

            actions.append([
                gate_type_dist.log_prob(gate_type),
                target_qubit_dist.log_prob(target_qubit),
                control_qubit_dist.log_prob(control_qubit)
            ])

            gate = get_gate(gate_type, target_qubit, control_qubit)
            logging.debug(f"\tAdding gate {gate}")

            if gate.__repr__() in GATE_LOG:
                GATE_LOG[gate.__repr__()] += 1
            else:
                GATE_LOG[gate.__repr__()] = 1

            state = update_state(state, gate)

            reward = compute_reward(state)
            rewards.append(reward)

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
        target_qubit_losses = []
        control_qubit_losses = []

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

            if min_entanglement_entropy(state) == 0.0:
                print(f"\n\tFound unentangled state. Breaking!")
                break

            gate_type_probs, control_qubit_probs, target_qubit_probs = agent(
                state)

            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_best(gate_type_dist)

            control_qubit_dist = Categorical(control_qubit_probs)
            control_qubit = sample_best(control_qubit_dist)

            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_best(target_qubit_dist)

            gate = get_gate(gate_type, target_qubit, control_qubit)

            if gate is not None:

                print(f"\tAdding gate {gate}")
                state = update_state(state, gate)

            else:
                print(f"\tBreaking episode at t={t}")
                break

        print(f"\nFinal state: {state}")
        print(f"\tFinal distance: {min_entanglement_entropy(state)}")
