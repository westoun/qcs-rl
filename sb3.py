
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from quasim import QuaSim, Circuit, get_unitary
from quasim.gates import (
    H,
    S,
    T,
    CX,
    IGate,
    Gate,
    CGate
)
from quasim.gates.utils import create_matrix, create_controlled_matrix
import random
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
import torch
from typing import List, Union

from utils.circuit import decomplexify_vector, create_random_circuit, update_state
from utils.metrics import min_entanglement_entropy

QUBIT_NUM = 2
GATE_TYPE_COUNT = 4  # Clifford gate set
MAX_STEPS = 10

# Constants of action space
H_GATE = 0
S_GATE = 1
T_GATE = 2
CX_GATE = 3
TERMINATE = 4


def create_random_state(gate_count: int = 5, qubit_num: int = 2) -> np.ndarray:
    # TODO: Move to utils.
    circuit = create_random_circuit(gate_count, qubit_num)

    simulator = QuaSim()
    simulator.evaluate([circuit])

    return circuit.state


def is_valid_action(gate_type: int, target_qubit: int, control_qubit: int) -> bool:
    if gate_type == CX_GATE:
        return target_qubit != control_qubit

    return True


def get_gate(gate_type: int, target_qubit: int, control_qubit: int) -> IGate:
    if gate_type == 0:
        return H(target_qubit)
    elif gate_type == 1:
        return S(target_qubit)
    elif gate_type == 2:
        return T(target_qubit)
    elif gate_type == 3:
        return CX(control_qubit, target_qubit)
    else:
        raise NotImplementedError()


def compute_reward(state, gate_type, invalid_action, step_limit_reached, gate_history, state_history) -> float:
    if invalid_action:
        return - 10

    if gate_type == TERMINATE:
        if min_entanglement_entropy(state) == 0:
            return 1000
        else:
            return -1

    # punish suggestion of the same gate twice in a row
    if len(gate_history) > 1 and gate_history[-1].__repr__() == gate_history[-2].__repr__():
        return - 5

    # case normal state update
    return -0.01


class StateSeparatorEnv(gym.Env):
    # metadata = {"render_modes": ["console"]}
    render_mode = "console"

    state: np.ndarray
    step_count: int

    state_history: List
    gate_history: List

    def __init__(self, seed: int = None):
        super().__init__()
        self.action_space = spaces.MultiDiscrete(
            [GATE_TYPE_COUNT + 1, QUBIT_NUM, QUBIT_NUM])
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(2 * 2 ** QUBIT_NUM, ), dtype=np.float64)

        if seed is not None:
            # TODO: set other seeds as well
            random.seed(seed)

    def step(self, action):
        gate_type, target_qubit, control_qubit = action

        self.step_count += 1
        step_limit_reached = self.step_count == MAX_STEPS

        invalid_action = not is_valid_action(
            gate_type, target_qubit, control_qubit)
        terminate = bool(invalid_action or gate_type == TERMINATE)

        if not terminate:
            gate = get_gate(gate_type, target_qubit, control_qubit)
            self.state = update_state(self.state, gate)

            self.gate_history.append(gate)
            self.state_history.append(self.state)

        observation = decomplexify_vector(self.state)
        reward = compute_reward(
            state=self.state,
            gate_type=gate_type,
            invalid_action=invalid_action,
            step_limit_reached=step_limit_reached,
            gate_history=self.gate_history,
            state_history=self.state_history
        )

        return observation, reward, terminate, step_limit_reached, {"state": self.state}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)

        # Start from entangled states to avoid getting stuck in
        # local optima always proposing none-gate.
        for _ in range(100):
            state = create_random_state(
                gate_count=20, qubit_num=QUBIT_NUM)

            if min_entanglement_entropy(state) > 0:
                break

        self.state = state
        self.step_count = 0
        self.gate_history = []
        self.state_history = [self.state]

        observation = decomplexify_vector(self.state)
        return observation, {"state": self.state}

    def render(self):
        if self.render_mode == "console":
            # TODO: Figure out what to put here.
            pass
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' has no implementation specified.")

    def close(self):
        pass


if __name__ == "__main__":

    env = StateSeparatorEnv(seed=1)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=1).learn(
        20000, log_interval=1000)

    test_count = 5
    for i in range(test_count):
        print(f"\nTest case {i}")

        obs, info = env.reset()
        print(f"Target state: {info['state']}")

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)

            print(f"\tStep {step + 1}")
            print(f"\tAction: {action}")

            obs, reward, termination_requested, step_limit_reached, info = env.step(
                action)

            done = termination_requested or step_limit_reached

            print(f"state= {info['state']}, reward= {reward}, done= {done}")

            env.render()
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                print(f"\n\tGoal reached! reward= {reward}")
                break
