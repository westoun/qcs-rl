

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


class GatePredictor(nn.Module):

    def __init__(self):
        super(GatePredictor, self).__init__()
        # times 2 because state is split into real and imaginary
        self.hidden_layer1 = nn.Linear(2 * 2 ** MAX_QUBITS, 16)
        self.hidden_layer2 = nn.Linear(16, 16)

        # Avoid none output for now.
        self.gate_type_pred = nn.Linear(16, GATE_TYPE_COUNT)

    def forward(self, state: np.ndarray):
        state = torch.tensor(decomplexify_vector(state), dtype=torch.float32)

        hidden1 = F.relu(self.hidden_layer1(state))
        hidden2 = F.relu(self.hidden_layer2(hidden1))

        gate_type_probs = F.softmax(self.gate_type_pred(hidden2), dim=-1)

        return gate_type_probs


class TargetQubitPredictor(nn.Module):

    def __init__(self):
        super(TargetQubitPredictor, self).__init__()

        self.hidden_layer1 = nn.Linear(
            2 * 2 ** MAX_QUBITS + GATE_TYPE_COUNT, 16)
        self.hidden_layer2 = nn.Linear(
            16, 16)

        self.target_qubit_pred = nn.Linear(16, MAX_QUBITS)

    def forward(self, state: np.ndarray, gate_one_hot: np.ndarray):
        state = decomplexify_vector(state)

        input_vector = torch.tensor(np.concatenate(
            [state, gate_one_hot]), dtype=torch.float32)

        hidden1 = F.relu(self.hidden_layer1(input_vector))
        hidden2 = F.relu(self.hidden_layer2(hidden1))

        target_qubit_probs = F.softmax(self.target_qubit_pred(hidden2), dim=-1)

        return target_qubit_probs


class ControlQubitPredictor(nn.Module):

    def __init__(self):
        super(ControlQubitPredictor, self).__init__()

        self.hidden_layer1 = nn.Linear(
            2 * 2 ** MAX_QUBITS + GATE_TYPE_COUNT + MAX_QUBITS, 16)
        self.hidden_layer2 = nn.Linear(
            16, 16)

        self.control_qubit_pred = nn.Linear(16, MAX_QUBITS)

    def forward(self, state: np.ndarray, gate_one_hot: np.ndarray, target_qubit_one_hot: np.ndarray):
        state = decomplexify_vector(state)

        input_vector = torch.tensor(np.concatenate(
            [state, gate_one_hot, target_qubit_one_hot]), dtype=torch.float32)

        hidden1 = F.relu(self.hidden_layer1(input_vector))
        hidden2 = F.relu(self.hidden_layer2(hidden1))

        control_qubit_probs = F.softmax(
            self.control_qubit_pred(hidden2), dim=-1)

        return control_qubit_probs


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(
            2 * 2 ** MAX_QUBITS, 16)

        self.value_output = nn.Linear(16, 1)

    def forward(self, state: np.ndarray):
        state = torch.tensor(decomplexify_vector(state), dtype=torch.float32)

        x = F.relu(self.input_layer(state))

        value = self.value_output(x)
        return value


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
    EPISODE_LENGTH = 5

    LOG_EVERY = 100
    MOVING_AVERAGE_WINDOW = 5

    SEPARABILITY_REWARD = 10
    PUNISHMENT_TERM = 2

    EPSILON = 0.1
    EPSILON_DECAY_RATE = 0.95

    gate_predictor = GatePredictor()
    target_qubit_predictor = TargetQubitPredictor()
    control_qubit_predictor = ControlQubitPredictor()

    critic = Critic()

    gate_optimizer = optim.Adam(gate_predictor.parameters(), lr=3e-2)
    target_qubit_optimizer = optim.Adam(
        target_qubit_predictor.parameters(), lr=3e-2)
    control_qubit_optimizer = optim.Adam(
        control_qubit_predictor.parameters(), lr=3e-2)

    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-2)

    episode_rewards = []
    moving_average_episode_rewards = []
    episodes = []

    GATE_LOG = {}

    for episode in range(EPISODES):
        logging.debug(f"Starting episode {episode}")

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
            EPSILON = EPSILON * EPSILON_DECAY_RATE

            from pprint import pprint
            pprint(GATE_LOG)
            GATE_LOG = {}

        actions = []
        values = []
        rewards = []

        # generate episode data
        for t in range(EPISODE_LENGTH):

            if min_entanglement_entropy(state) == 0:
                logging.debug(f"\tBreaking episode at t={t}.")
                break

            value = critic(state)
            values.append(value)

            gate_type_probs = gate_predictor(state)
            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_epsilon_greedy(gate_type_dist, epsilon=EPSILON)
            gate_one_hot = one_hot_encode(GATE_TYPE_COUNT, gate_type)

            target_qubit_probs = target_qubit_predictor(state, gate_one_hot)
            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_epsilon_greedy(
                target_qubit_dist, epsilon=EPSILON)
            target_qubit_one_hot = one_hot_encode(MAX_QUBITS, target_qubit)

            # case: CX gate
            if gate_type == 3:
                control_qubit_probs = control_qubit_predictor(
                    state, gate_one_hot, target_qubit_one_hot)
                control_qubit_dist = Categorical(control_qubit_probs)
                control_qubit = sample_epsilon_greedy(
                    control_qubit_dist, epsilon=EPSILON)

                actions.append([
                    gate_type_dist.log_prob(gate_type),
                    target_qubit_dist.log_prob(target_qubit),
                    control_qubit_dist.log_prob(control_qubit)
                ])

                if control_qubit == target_qubit:
                    rewards.append(-PUNISHMENT_TERM * EPISODE_LENGTH / 2)
                    logging.debug(f"\tBreaking because of invalid gate.")

                    if "invalid" in GATE_LOG:
                        GATE_LOG["invalid"] += 1
                    else:
                        GATE_LOG["invalid"] = 1

                    break

                gate = get_gate(gate_type, target_qubit, control_qubit)

            else:
                actions.append([
                    gate_type_dist.log_prob(gate_type),
                    target_qubit_dist.log_prob(target_qubit)
                ])
                gate = get_gate(gate_type, target_qubit)

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

        value_losses = []

        for log_probs, value, discounted_reward in zip(actions, values, discounted_rewards):
            advantage = discounted_reward - value.item()
            value_losses.append(F.smooth_l1_loss(
                value, torch.tensor([discounted_reward])))

            gate_type_losses.append(
                -log_probs[0] * advantage
            )

            target_qubit_losses.append(
                -log_probs[1] * advantage
            )

            if len(log_probs) == 3:
                control_qubit_losses.append(
                    -log_probs[2] * advantage
                )

        critic_optimizer.zero_grad()
        torch.stack(value_losses).sum().backward(retain_graph=True)
        critic_optimizer.step()

        gate_optimizer.zero_grad()
        torch.stack(gate_type_losses).sum().backward(retain_graph=True)
        gate_optimizer.step()

        target_qubit_optimizer.zero_grad()
        torch.stack(target_qubit_losses).sum().backward(retain_graph=True)
        target_qubit_optimizer.step()

        if len(control_qubit_losses) > 0:
            control_qubit_optimizer.zero_grad()
            torch.stack(control_qubit_losses).sum().backward(retain_graph=True)
            control_qubit_optimizer.step()

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

            gate_type_probs = gate_predictor(state)
            gate_type_dist = Categorical(gate_type_probs)
            gate_type = sample_best(gate_type_dist)
            gate_one_hot = one_hot_encode(GATE_TYPE_COUNT, gate_type)

            target_qubit_probs = target_qubit_predictor(state, gate_one_hot)
            target_qubit_dist = Categorical(target_qubit_probs)
            target_qubit = sample_best(
                target_qubit_dist)
            target_qubit_one_hot = one_hot_encode(MAX_QUBITS, target_qubit)

            # case: CX gate
            if gate_type == 3:
                control_qubit_probs = control_qubit_predictor(
                    state, gate_one_hot, target_qubit_one_hot)
                control_qubit_dist = Categorical(control_qubit_probs)
                control_qubit = sample_best(
                    control_qubit_dist)

                gate = get_gate(gate_type, target_qubit, control_qubit)

            else:
                gate = get_gate(gate_type, target_qubit)

            print(f"\tAdding gate {gate}")

            state = update_state(state, gate)

        print(f"\nFinal state: {state}")
        print(f"\tFinal distance: {min_entanglement_entropy(state)}")
