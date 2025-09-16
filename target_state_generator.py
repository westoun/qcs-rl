
import hashlib
from math import floor
import numpy as np
from random import sample, randint
from typing import Set, List

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

from utils.circuit import decomplexify_vector, create_random_circuit, update_state
from utils.metrics import min_entanglement_entropy


def hash(my_string: str) -> str:
    # Use custom hash function because built-in method
    # uses random seed as additional security feature.
    hash_obj = hashlib.sha256(my_string.encode())
    hex_hash = hash_obj.hexdigest()
    return str(hex_hash)


class TargetStateGenerator:
    qubit_num: int
    gate_count: int
    train_test_split: float
    retries_per_circuit: int
    max_pool_size: int

    train_pool: List
    test_pool: List

    # Use separate unused pool to avoid returning
    # duplicate states by chance alone.
    _unused_train_pool: List
    _unused_test_pool: List

    def __init__(self, qubit_num: int, gate_count: int, max_pool_size: int, train_test_split: float = 0.3, retries_per_circuit: int = 100):
        self.qubit_num = qubit_num
        self.gate_count = gate_count
        self.train_test_split = train_test_split
        self.retries_per_circuit = retries_per_circuit
        self.max_pool_size = max_pool_size

        self.build_pool(qubit_num=qubit_num, gate_count=gate_count,
                        max_pool_size=max_pool_size)

    def build_pool(self, qubit_num: int, gate_count: int, max_pool_size: int = None) -> None:
        if max_pool_size is None:
            max_pool_size = self.max_pool_size

        pool = []
        for _ in range(max_pool_size):

            # Start from entangled states to avoid getting stuck in
            # local optima always proposing none-gate.
            for _ in range(self.max_pool_size):
                state = self.create_random_state(
                    qubit_num=qubit_num,
                    gate_count=gate_count)

                if min_entanglement_entropy(state) > 0:
                    break

            pool.append(state)

        # remove duplicates
        unique_pool = []
        added_states = set()
        for state in pool:
            state_hash = hash(np.array2string(state, precision=3))

            if state_hash in added_states:
                continue

            unique_pool.append(state)
            added_states.add(state_hash)

        train_size = floor(len(unique_pool) * (1 - self.train_test_split))

        self.train_pool = unique_pool[:train_size]
        self.test_pool = unique_pool[train_size:]

        self._unused_train_pool = self.train_pool[:]
        self._unused_test_pool = self.test_pool[:]

    def create_random_state(self, qubit_num: int = 2, gate_count: int = 5) -> np.ndarray:
        circuit = create_random_circuit(gate_count, qubit_num)

        simulator = QuaSim()
        simulator.evaluate([circuit])

        return circuit.state

    def get_state(self, eval: bool = False) -> np.ndarray:
        if eval:
            if len(self._unused_test_pool) == 0:
                self._unused_test_pool = self.test_pool[:]

            target_idx = randint(0, len(self._unused_test_pool) - 1)
            return self._unused_test_pool.pop(target_idx)

        else:
            if len(self._unused_train_pool) == 0:
                self._unused_train_pool = self.train_pool[:]

            target_idx = randint(0, len(self._unused_train_pool) - 1)
            return self._unused_train_pool.pop(target_idx)
