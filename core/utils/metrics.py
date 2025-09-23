

from math import log2, floor
import numpy as np
from pennylane.math import vn_entanglement_entropy, dm_from_state_vector
from typing import List, Tuple


def enumerate_qubit_groupings(qubit_num: int) -> List[Tuple[List[int], List[int]]]:
    # Underlying question: how to get all possibilities of spliting a set in two?
    # Solution based on https://stackoverflow.com/a/6999554

    qubits = [qubit for qubit in range(qubit_num)]

    qubit_groupings = []

    solution_count = 2**(qubit_num - 1) - 1

    for solution_i in range(1, solution_count + 1):

        indices0 = []
        indices1 = []

        solution_encoding = "0" * \
            (len(bin(solution_count + 1)) -
             len(bin(solution_i)) - 1) + bin(solution_i)[2:]

        for qubit_i, char in enumerate(solution_encoding):
            if char == "1":
                indices0.append(qubits[qubit_i])
            else:
                indices1.append(qubits[qubit_i])

        indices1.append(qubits[-1])

        qubit_groupings.append((indices0, indices1))

    return qubit_groupings


def min_entanglement_entropy(state: np.ndarray) -> float:
    density_matrix = dm_from_state_vector(state)

    qubit_num = int(log2(len(state)))

    current_minimum = np.inf
    for indices0, incides1 in enumerate_qubit_groupings(qubit_num):
        entanglement_entropy = vn_entanglement_entropy(
            density_matrix, indices0=indices0, indices1=incides1)

        if entanglement_entropy < 1e-8:
            return 0

        if entanglement_entropy < current_minimum:
            current_minimum = entanglement_entropy

    return current_minimum
