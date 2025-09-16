import numpy as np
from random import random, randint
import torch
import torch.nn as nn
from torch.distributions import Categorical


def sample_epsilon_greedy(dist: Categorical, epsilon: float = 0.01) -> torch.Tensor:
    if random() < (1 - epsilon):
        return sample_best(dist)
    else:
        action_idx = randint(0, len(dist.probs) - 1)
        # Workaround to ensure that return formats match
        return torch.LongTensor([action_idx])[0]


def sample(dist: Categorical) -> torch.Tensor:
    return dist.sample()


def sample_best(dist: Categorical) -> torch.Tensor:
    return dist.probs.argmax()
