from datetime import datetime
import json
import numpy as np
from numpy import random as np_random
import os
import random
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import set_random_seed
import torch
from typing import List, Union, Tuple

from environment import StateSeparatorEnv
from target_state_generator import TargetStateGenerator


def run_experiment(
        qubit_num: int,
        gate_count: int,
        max_pool_size: int,
        seed: int,
        max_steps: int,
        total_timesteps: int,
        log_dir: str = "logs",
        train_test_split: float = 0.3,
        n_eval_episodes: int = 1000,
        eval_freq: int = 50_000,
        log_interval: int = 1_000,
        tag: str = "base"
) -> Tuple[OnPolicyAlgorithm, StateSeparatorEnv]:
    # do not include seed value in target path
    # so that multiple seeds of same setup are
    # stored in same path.
    log_dir = f"{log_dir}/{qubit_num}q_{tag}"
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "start_time": str(datetime.now()),
        "common_params": {
            "qubit_num": qubit_num,
            "max_steps": max_steps,
            "seed": seed,
            "tag": tag,
        },
        "state_generator_params": {
            "gate_count": gate_count,
            "max_pool_size": max_pool_size,
            "train_test_split": train_test_split,
        },
        "train_params": {
            "total_timesteps": total_timesteps,
            "log_interval": log_interval,
        },
        "test_env_params": {
            "n_eval_episodes": n_eval_episodes,
            "eval_freq": eval_freq,
        },
    }
    with open(f"{log_dir}/config_{seed}.json", "w") as config_file:
        json.dump(config, config_file)

    random.seed(seed)
    np_random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed)

    state_generator = TargetStateGenerator(
        qubit_num=qubit_num, gate_count=gate_count, max_pool_size=max_pool_size, train_test_split=train_test_split)

    raw_env = StateSeparatorEnv(qubit_num=qubit_num, max_steps=max_steps,
                                state_generator=state_generator, eval=False)
    check_env(raw_env)

    env = Monitor(raw_env, f"{log_dir}/train_{seed}")

    raw_eval_env = StateSeparatorEnv(
        qubit_num=qubit_num, max_steps=max_steps, state_generator=state_generator, eval=True)
    check_env(raw_eval_env)

    eval_env = Monitor(raw_eval_env, f"{log_dir}/test_{seed}")

    eval_callback = EvalCallback(
        eval_env, log_path=log_dir, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, deterministic=True, render=False
    )

    model = PPO("MlpPolicy", env, verbose=1, seed=seed, n_steps=5_000)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval,
                progress_bar=True, callback=eval_callback)

    return model, raw_eval_env


def test_model(model: OnPolicyAlgorithm, env: StateSeparatorEnv, max_steps: int, n: int = 5) -> None:
    for i in range(n):
        print(f"\nTest case {i + 1}")

        obs, info = env.reset()
        print(f"Target state: {info['state']}")

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)

            print(f"\tStep {step + 1}")
            print(f"\tAction: {action}")

            obs, reward, found_solution, step_limit_reached, info = env.step(
                action)

            done = found_solution or step_limit_reached

            print(f"state= {info['state']}, reward= {reward}, done= {done}")

            env.render()
            if found_solution:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                print(f"\n\tGoal reached! reward= {reward}")
                break

            if step_limit_reached:
                print(f"\n\tBreaking because max steps exceeded.")
                break


if __name__ == "__main__":

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=10,
        max_pool_size=20_000,
        max_steps=10,
        total_timesteps=2_000_000,
        eval_freq=5_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    test_model(model, eval_env, max_steps=10)
