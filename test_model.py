
import numpy as np
from numpy import kron, array2string
from numpy import random as np_random
from numpy.linalg import eig, det, norm, svd
from numpy.linalg.linalg import SVDResult
from pennylane.math import reduce_statevector, dm_from_state_vector, partial_trace, vn_entropy
from pprint import pprint
from quasim import QuaSim
import random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
import torch
from typing import List, Dict
import math

from core.utils.circuit import create_random_circuit, decomplexify_vector, update_state
from core.utils.metrics import min_entanglement_entropy, get_best_qubit_grouping
from core.environment import get_gate, is_valid_action


def get_mapping_dict(qubit_order: List[int]) -> Dict:
    mapping_dict = {}

    for old_state_idx in range(2 ** len(qubit_order)):

        old_bit_string = bin(old_state_idx)[2:].zfill(len(qubit_order))

        new_state_idx = 0

        for bit_i, bit_value in enumerate(old_bit_string):

            if bit_value == "0":
                continue

            new_state_idx += 2 ** (len(qubit_order) -
                                   1 - qubit_order.index(bit_i))

        mapping_dict[new_state_idx] = old_state_idx

    return mapping_dict


def swap_qubit_order(state: np.ndarray, qubit_order: List[int]) -> np.ndarray:
    assert math.log2(len(state)) == len(qubit_order)

    mapping_dict = get_mapping_dict(qubit_order)

    new_state = np.zeros(shape=state.shape, dtype=state.dtype)

    for new_i, old_i in mapping_dict.items():
        new_state[new_i] = state[old_i]

    return new_state


def invert_swapping_order(qubit_order: List[int]) -> List[int]:
    inverse_order = [0] * len(qubit_order)

    for i, target_position in enumerate(qubit_order):
        inverse_order[target_position] = i

    return inverse_order


def check_state_equivalence(state1: np.ndarray, state2: np.ndarray) -> bool:

    factors = []
    for val1, val2 in zip(state1, state2):

        factor = abs((val1 + 0.0001) / (val2 + 0.0001))
        factors.append(factor)

    violating_factors = [
        factor for factor in factors if factor <= 0.9 or factor >= 1.1
    ]

    print()
    if len(violating_factors) == 0:
        print("\033[92mStates are equivalent.\033[0m")
        return True

    else:
        print("\033[91mStates are not equivalent.\033[0m")

        print(f"{len(violating_factors)} of {len(state1)} cells don't match.")
        print(
            f"min factor: {min(violating_factors)}, max factor: {max(violating_factors)}")
        return False


def sanity_check_decomposition(decomposition: SVDResult) -> None:
    length1 = 0
    for val in decomposition.U[0]:
        length1 += abs(val) ** 2

    length2 = 0
    for val in decomposition.Vh[0]:
        length2 += abs(val) ** 2

    print()
    if not (length1 >= 0.99 and length1 <= 1.01 and length2 >= 0.99 and length2 <= 1.01):
        print("\033[91mOne or more eigenvectors have the wrong length.\033[0m")
    elif not (abs(decomposition.S[0]) >= 0.99 and abs(decomposition.S[0]) <= 1.01):
        print("\033[91mFirst eigenvalue does not have the right length.\033[0m")
    else:
        print("\033[96mDecomposition passes sanity checks.\033[0m")


EXAMPLE_COUNT = 20

QUBIT_NUM = 4
SEED = 1
GATE_COUNT = 25
MAX_TRIES = 10_000
MODEL_PATH = "logs/4q_25g_base/1/model.zip"

if SEED is not None:
    random.seed(SEED)
    np_random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    set_random_seed(SEED)

model = PPO.load(MODEL_PATH)
simulator = QuaSim()

for example_i in range(EXAMPLE_COUNT):

    print("\n====================")
    print(f"Example {example_i}:")
    print("====================")

    for _ in range(MAX_TRIES):
        circuit = create_random_circuit(
            qubit_num=QUBIT_NUM, gate_count=GATE_COUNT)
        simulator.evaluate([circuit])
        state = circuit.state

        if min_entanglement_entropy(state) > 0:
            break
    else:
        raise ValueError(
            f"Failed to construct entangled state within {MAX_TRIES} attempts.")

    state_history = [state]
    gate_history = []

    for step in range(GATE_COUNT):
        observation = decomplexify_vector(state)

        action, _ = model.predict(observation, deterministic=True)
        gate_type, target_qubit, control_qubit = action

        # TA: how to deal with invalid actions here?

        if not is_valid_action(gate_type, target_qubit, control_qubit):
            print("Encountered invalid action in deterministic model.")
            break

        gate = get_gate(gate_type, target_qubit, control_qubit)
        state = update_state(state, gate)

        state_history.append(state)
        gate_history.append(gate)

        if min_entanglement_entropy(state) == 0:
            print("Found solution!")
            # print(circuit)

            # print(array2string(state_history[0],
            #       precision=3, suppress_small=True))
            # for gate, state in zip(gate_history, state_history[1:]):
            #     print(f"\nx {gate} =")
            #     print(state)

            group1, group2 = get_best_qubit_grouping(state)

            formatted_state = swap_qubit_order(state, group1 + group2)

            print("Qubit grouping", group1, group2)

            # print("Qubit grouping of formatted state",
            #       get_best_qubit_grouping(formatted_state))

            pprint(get_mapping_dict(group1 + group2))

            tmp_mtx = formatted_state.reshape((2**len(group1), 2**len(group2)))

            decomposition = svd(tmp_mtx)

            sanity_check_decomposition(decomposition)

            product_state1 = decomposition.U[0]
            product_state2 = decomposition.Vh[0]

            print("\nProduct state 1")
            print(array2string(product_state1, suppress_small=True, precision=3))

            print("Product state 2")
            print(array2string(product_state2, suppress_small=True, precision=3))

            reconstructed_state = kron(product_state1, product_state2)
            reconstructed_state = swap_qubit_order(
                reconstructed_state, invert_swapping_order(group1 + group2))

            print("\nOriginal state")
            print(array2string(state, suppress_small=True, precision=3))

            print("Reconstructed state")
            print(array2string(reconstructed_state,
                  suppress_small=True, precision=3))

            check_state_equivalence(state, reconstructed_state)

            print(
                array2string(state / reconstructed_state,
                             suppress_small=True, precision=2)
            )

            break

            reduced_dm1 = reduce_statevector(state, indices=group1)
            reduced_dm2 = reduce_statevector(state, indices=group2)

            # VN Entropy of reduced dms is very low.

            reduced_state1 = eig(reduced_dm1)
            reduced_state2 = eig(reduced_dm2)

            eigenvector1 = None
            print("\nGroup 1 eigenvectors:")
            for eigenvector, eigenvalue in zip(reduced_state1.eigenvectors, reduced_state1.eigenvalues):
                if abs(eigenvalue) >= 0.98 and abs(eigenvalue) <= 1.02:
                    print(eigenvector)

                    length = 0
                    for value in eigenvector:
                        length += abs(value) ** 2

                    print(length)

                    eigenvector1 = eigenvector

            eigenvector2 = None
            print("\nGroup 2 eigenvectors:")
            for eigenvector, eigenvalue in zip(reduced_state2.eigenvectors, reduced_state2.eigenvalues):
                if abs(eigenvalue) >= 0.98 and abs(eigenvalue) <= 1.02:
                    print(eigenvector)
                    eigenvector2 = eigenvector

                    length = 0
                    for value in eigenvector:
                        length += abs(value) ** 2

                    print(length)
                    eigenvector2 = eigenvector2 / length

            reconstructed_state = kron(eigenvector2, eigenvector1)

            print(array2string(state, suppress_small=True, precision=1))
            print(array2string(reconstructed_state,
                  suppress_small=True, precision=1))

            # print("\nFull state:", array2string(full_state, precision=3, suppress_small=True))

            # print(norm(eigenvector1))
            # print(norm(eigenvector2))

            # for gate in reversed(gate_history):
            #     full_state = update_state(full_state, gate)
            #     print("")
            #     print(array2string(full_state, precision=3, suppress_small=True))
            #     print(norm(full_state))

            break

    else:
        print("Did not find solution within step limit.")
