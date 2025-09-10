
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
from stable_baselines3.common.env_checker import check_env
import torch
from typing import List, Union

from utils.circuit import decomplexify_vector, create_random_circuit, update_state
from utils.metrics import min_entanglement_entropy

QUBIT_NUM = 2
GATE_TYPE_COUNT = 4  # Clifford gate set
MAX_STEPS = 8

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


def compute_reward(state: np.ndarray, invalid_gate: bool = False) -> float:
    if invalid_gate:
        return - 20

    distance = min_entanglement_entropy(state)

    if distance == 0:
        return 10
    else:
        return - 0.5


class StateSeparatorEnv(gym.Env):
    # metadata = {"render_modes": ["console"]}
    render_mode = "console"

    state: np.ndarray
    step_count: int

    def __init__(self, seed: int = None):
        super().__init__()
        self.action_space = spaces.MultiDiscrete(
            [GATE_TYPE_COUNT, QUBIT_NUM, QUBIT_NUM])
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(2 * 2 ** QUBIT_NUM, ), dtype=np.float64)

        if seed is not None:
            # TODO: set other seeds as well
            random.seed(seed)

    def step(self, action):
        gate_type, target_qubit, control_qubit = action

        self.step_count += 1

        step_limit_reached = self.step_count == MAX_STEPS

        # Need to cast because otherwise evaluates to numpy.bool
        termination_requested = bool(gate_type == TERMINATE)

        if not is_valid_action(gate_type, target_qubit, control_qubit):
            reward = compute_reward(self.state, invalid_gate=True)

        else:
            gate = get_gate(gate_type, target_qubit, control_qubit)

            self.state = update_state(self.state, gate)
            reward = compute_reward(self.state)

        observation = decomplexify_vector(self.state)
        return observation, reward, termination_requested, step_limit_reached, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)

        # TODO: Move to separate function
        # Start from entangled states to avoid getting stuck in
        # local optima always proposing none-gate.
        for _ in range(100):
            state = create_random_state(
                gate_count=20, qubit_num=QUBIT_NUM)

            if min_entanglement_entropy(state) > 0:
                break

        self.state = state
        self.step_count = 0

        observation = decomplexify_vector(self.state)
        return observation, {}

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

    env = StateSeparatorEnv()

    check_env(env)
