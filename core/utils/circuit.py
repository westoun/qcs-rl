

import math
import numpy as np
from random import choice, sample, randint
import re
from typing import List

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

GATE_TYPES = [
    "H", "S", "T", "CX"
]


def state_to_string(state: np.ndarray, precision: int = 5) -> str:
    state = np.array2string(state, precision=precision)
    state = state.replace("\n", "")
    state = re.sub(' +', ' ', state)
    return state


def decomplexify_vector(x: np.ndarray) -> np.ndarray:
    return np.array(x.real.tolist() + x.imag.tolist())


def create_random_circuit(gate_count: int, qubit_num: int = 2) -> Circuit:
    quasim_circuit = Circuit(qubit_num)

    for _ in range(gate_count):
        gate_type = choice(GATE_TYPES)

        if gate_type == "H":
            target_qubit = randint(0, qubit_num - 1)
            quasim_circuit.apply(H(target_qubit))

        elif gate_type == "S":
            target_qubit = randint(0, qubit_num - 1)
            quasim_circuit.apply(S(target_qubit))

        elif gate_type == "T":
            target_qubit = randint(0, qubit_num - 1)
            quasim_circuit.apply(T(target_qubit))

        elif gate_type == "CX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quasim_circuit.apply(CX(control_qubit, target_qubit))

        else:
            raise NotImplementedError()

    return quasim_circuit


def create_random_states(n: int, gate_count: int = 5, qubit_num: int = 2) -> List[np.ndarray]:
    circuits = [
        create_random_circuit(gate_count, qubit_num) for _ in range(n)
    ]

    simulator = QuaSim()
    simulator.evaluate(circuits)

    states = [circuit.state for circuit in circuits]
    return states


def update_state(state: np.ndarray, gate: IGate) -> np.ndarray:

    # workaround because quasim does not support simulation based on an input
    # state as of now.

    qubit_num = int(math.log(len(state), 2))

    if issubclass(gate.__class__, Gate):
        unitary = create_matrix(gate.matrix, gate.target_qubit, qubit_num)
        return np.matmul(unitary, state)
    elif issubclass(gate.__class__, CGate):
        unitary = create_controlled_matrix(
            gate.matrix, gate.control_qubit, gate.target_qubit, qubit_num)
        return np.matmul(unitary, state)
    else:
        raise NotImplementedError()
