import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
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
from typing import List, Dict

from core.utils.circuit import decomplexify_vector, state_to_string, update_state
from core.utils.metrics import min_entanglement_entropy, get_best_qubit_grouping
from core.state_generator import TargetStateGenerator

GATE_TYPE_COUNT = 4  # Clifford gate set

# Constants of action space
H_GATE = 0
S_GATE = 1
T_GATE = 2
CX_GATE = 3


def is_valid_action(gate_type: int, target_qubit: int, control_qubit: int) -> bool:
    if gate_type == CX_GATE:
        return target_qubit != control_qubit

    return True


def get_gate(gate_type: int, target_qubit: int, control_qubit: int) -> IGate:
    if gate_type == H_GATE:
        return H(target_qubit)
    elif gate_type == S_GATE:
        return S(target_qubit)
    elif gate_type == T_GATE:
        return T(target_qubit)
    elif gate_type == CX_GATE:
        return CX(control_qubit, target_qubit)
    else:
        raise NotImplementedError()


def compute_reward(state, gate_type, invalid_action, step_limit_reached, gate_history, state_history) -> float:
    # Whenever this function changes, make sure to check the cutoff values used for
    # success computation in evaluation.ipynb and curriculum learning callback.

    if invalid_action:
        return - 20

    if min_entanglement_entropy(state) == 0:  # is state separable?
        return 1000

    # punish suggestion of the same gate twice in a row
    if len(gate_history) > 1 and gate_history[-1].__repr__() == gate_history[-2].__repr__():
        return - 10

    # case normal state update
    return -0.01


class StateSeparatorEnv(gym.Env):
    state_generator: TargetStateGenerator
    eval: bool

    qubit_num: int
    max_steps: int

    state: np.ndarray
    step_count: int

    state_history: List
    gate_history: List

    def __init__(self, qubit_num: int, max_steps: int, state_generator: TargetStateGenerator, eval: bool = False):
        super().__init__()

        self.state_generator = state_generator
        self.eval = eval

        self.qubit_num = qubit_num
        self.max_steps = max_steps

        self.action_space = spaces.MultiDiscrete(
            [GATE_TYPE_COUNT, self.qubit_num, self.qubit_num])
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(2 * 2 ** self.qubit_num, ), dtype=np.float64)

    def step(self, action):
        gate_type, target_qubit, control_qubit = action

        self.step_count += 1
        step_limit_reached = self.step_count == self.max_steps

        invalid_action = not is_valid_action(
            gate_type, target_qubit, control_qubit)

        if not invalid_action:
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

        found_solution = bool(min_entanglement_entropy(self.state) == 0)

        info_dict = {"state": self.state,
                     "found_solution": found_solution,
                     "best_qubit_grouping": None,
                     "start_state": state_to_string(self.state_history[0]),
                     "gate_sequence": [
                         gate.__repr__() for gate in self.gate_history
                     ]}

        if found_solution:
            info_dict["best_qubit_grouping"] = get_best_qubit_grouping(
                self.state)

        return observation, reward, found_solution, step_limit_reached, info_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        state = self.state_generator.get_state(eval=self.eval)
        self.state = state

        self.step_count = 0
        self.gate_history = []
        self.state_history = [self.state]

        observation = decomplexify_vector(self.state)
        return observation, {"state": self.state}

    def render(self):
        pass

    def close(self):
        pass

    @property
    def params(self) -> Dict:
        return {
            "qubit_num": self.qubit_num,
            "max_steps": self.max_steps,
            "eval": self.eval
        }
