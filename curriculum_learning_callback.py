#!/usr/bin/env python3

import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import List

from environment import StateSeparatorEnv
from target_state_generator import TargetStateGenerator


class CurriculumLearningCallback(BaseCallback):

    log_path: str
    state_generator: TargetStateGenerator
    envs: List[StateSeparatorEnv]
    n_eval_episodes: int
    succ_pct_threshold: float
    max_gate_count: int

    def __init__(self, log_path: str, state_generator: TargetStateGenerator, envs: List[StateSeparatorEnv],
                 n_eval_episodes: int, succ_pct_threshold: float = 0.8, max_gate_count: int = 100, verbose: int = 0):
        super().__init__(verbose)

        assert len(envs) > 0

        self.log_path = log_path
        self.state_generator = state_generator
        self.envs = envs
        self.n_eval_episodes = n_eval_episodes
        self.succ_pct_threshold = succ_pct_threshold
        self.max_gate_count = max_gate_count

    def _on_step(self) -> bool:

        # get and compute success pct.
        succ_pct = self._compute_succ_pct(self.log_path)

        if succ_pct >= self.succ_pct_threshold and self.state_generator.gate_count < self.max_gate_count:

            print("Increasing gate count and max steps.")

            self.state_generator.gate_count += 1
            self.state_generator.build_pool()

            for env in self.envs:
                env.max_steps += 1

        # Must return true or training is aborted.
        return True

    def _compute_succ_pct(self, path: str) -> float:
        monitor_df = pd.read_csv(path, skiprows=1)

        assert len(monitor_df) % self.n_eval_episodes == 0

        last_eval_df = monitor_df[-self.n_eval_episodes:]
        last_eval_succ_df = last_eval_df[last_eval_df["l"] < self.envs[0].max_steps
                                         ]

        return len(last_eval_succ_df) / len(last_eval_df)
