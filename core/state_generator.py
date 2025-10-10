
import hashlib
from math import floor
import numpy as np
from random import sample, randint
from typing import Set, List, Dict

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

from core.utils.circuit import decomplexify_vector, create_random_state, update_state
from core.utils.metrics import min_entanglement_entropy


class TargetStateGenerator:
    qubit_num: int
    gate_count: int
    train_test_split: float
    targeted_pool_size: int
    max_generation_tries: int

    train_pool: List
    test_pool: List

    # Use separate unused pool to avoid returning
    # duplicate states by chance alone.
    _unused_train_pool: List
    _unused_test_pool: List

    def __init__(self, qubit_num: int, gate_count: int, targeted_pool_size: int, train_test_split: float = 0.3, max_generation_tries: int = 1_000_000):
        assert targeted_pool_size <= max_generation_tries

        self.qubit_num = qubit_num
        self.gate_count = gate_count
        self.train_test_split = train_test_split
        self.targeted_pool_size = targeted_pool_size
        self.max_generation_tries = max_generation_tries

        self.build_pool()

    def build_pool(self) -> None:
        pool: List[np.ndarray] = []
        processed_states = set()
        for _ in range(self.max_generation_tries):
            if len(pool) >= self.targeted_pool_size:
                break

            state = create_random_state(
                qubit_num=self.qubit_num,
                gate_count=self.gate_count)

            # if a low precision value is chosen, differences arise
            # when the same setup (same seed values) is run twice
            # within the same process. I believe, these issues are
            # due to some non-deterministic aspect of float rounding
            # in numpy, but don't know for sure.
            state_hash = hash(np.array2string(
                state, precision=64, suppress_small=True))

            if state_hash in processed_states:
                continue

            processed_states.add(state_hash)

            # Start from entangled states to avoid getting stuck in
            # local optima always proposing none-gate.
            if min_entanglement_entropy(state) == 0:
                continue

            pool.append(state)

        train_size = floor(len(pool) * (1 - self.train_test_split))

        self.train_pool = pool[:train_size]
        self.test_pool = pool[train_size:]

        self._unused_train_pool = self.train_pool.copy()
        self._unused_test_pool = self.test_pool.copy()

    def get_state(self, eval: bool = False) -> np.ndarray:
        if eval:
            if len(self._unused_test_pool) == 0:
                self._unused_test_pool = self.test_pool.copy()

            target_idx = randint(0, len(self._unused_test_pool) - 1)
            return self._unused_test_pool.pop(target_idx)

        else:
            if len(self._unused_train_pool) == 0:
                self._unused_train_pool = self.train_pool.copy()

            target_idx = randint(0, len(self._unused_train_pool) - 1)
            return self._unused_train_pool.pop(target_idx)

    @property
    def params(self) -> Dict:
        return {
            "qubit_num": self.qubit_num,
            "gate_count": self.gate_count,
            "train_test_split": self.train_test_split,
            "targeted_pool_size": self.targeted_pool_size,
            "max_generation_tries": self.max_generation_tries,
            "train_pool_size": len(self.train_pool),
            "test_pool_size": len(self.test_pool)
        }
