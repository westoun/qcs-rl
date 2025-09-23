#!/usr/bin/env python3

from datetime import datetime
import json
from numpy import random as np_random
import os
import random
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import set_random_seed
import torch
from typing import List, Union, Tuple

from environment import StateSeparatorEnv
from target_state_generator import TargetStateGenerator
from curriculum_learning_callback import CurriculumLearningCallback

# Show progressbar if experiment is run locally (requirements.txt).
# Don't show if experiment is run in docker (experiment_requirements.txt).
try:
    import tqdm
    SHOW_PROGRESSBAR = True
except ImportError:
    SHOW_PROGRESSBAR = False


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def run_experiment(
        qubit_num: int,
        gate_count: int,
        targeted_pool_size: int,
        seed: int,
        max_steps: int,
        total_timesteps: int,
        log_dir: str = "logs",
        train_test_split: float = 0.3,
        max_generation_tries: int = 1_000_000,
        n_eval_episodes: int = 1_000,
        eval_freq: int = 50_000,
        log_interval: int = 1_000,
        tag: str = "base",
        curriculum_learning: bool = False
) -> Tuple[OnPolicyAlgorithm, StateSeparatorEnv]:
    # do not include seed value in target path
    # so that multiple seeds of same setup are
    # stored in same path.
    log_dir = f"{log_dir}/{qubit_num}q_{gate_count}g_{tag}/{seed}"
    os.makedirs(log_dir, exist_ok=True)

    config_target_path = f"{log_dir}/config.json"

    config = {
        "start_time": str(datetime.now()),
        "common_params": {
            "qubit_num": qubit_num,
            "gate_count": gate_count,
            "seed": seed,
            "tag": tag,
            "curriculum_learning": curriculum_learning
        },
        "state_generator_params": None,
        "train_params": {
            "total_timesteps": total_timesteps,
            "log_interval": log_interval,
        },
        "eval_params": {
            "n_eval_episodes": n_eval_episodes,
            "eval_freq": eval_freq,
        },
        "train_env_params": None,
        "eval_env_params": None,
        "curriculum_learning_params": None
    }

    random.seed(seed)
    np_random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed)

    state_generator = TargetStateGenerator(
        qubit_num=qubit_num, gate_count=gate_count, targeted_pool_size=targeted_pool_size, train_test_split=train_test_split, max_generation_tries=max_generation_tries)
    config["state_generator_params"] = state_generator.params

    raw_env = StateSeparatorEnv(qubit_num=qubit_num, max_steps=max_steps,
                                state_generator=state_generator, eval=False)
    config["train_env_params"] = raw_env.params
    check_env(raw_env)

    env = Monitor(raw_env, f"{log_dir}/train")

    raw_eval_env = StateSeparatorEnv(
        qubit_num=qubit_num, max_steps=max_steps, state_generator=state_generator, eval=True)
    config["eval_env_params"] = raw_eval_env.params
    check_env(raw_eval_env)

    eval_env = Monitor(raw_eval_env, f"{log_dir}/test")

    if curriculum_learning is True:
        callback_after_eval = CurriculumLearningCallback(
            log_path=f"{log_dir}/test.monitor.csv", state_generator=state_generator,
            envs=[
                raw_env, raw_eval_env], n_eval_episodes=n_eval_episodes, succ_pct_threshold=0.8,
            max_gate_count=25
        )
        config["curriculum_learning_params"] = callback_after_eval.params
    else:
        callback_after_eval = None

    eval_callback = EvalCallback(
        eval_env, log_path=log_dir, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, deterministic=True, render=False, callback_after_eval=callback_after_eval
    )

    model = PPO("MlpPolicy", env, verbose=1, seed=seed, n_steps=5_000)

    config["common_params"]["model"] = str(model.policy)

    save_to_json(config, config_target_path)

    model.learn(total_timesteps=total_timesteps, log_interval=log_interval,
                progress_bar=SHOW_PROGRESSBAR, callback=eval_callback)

    model.save(f"{log_dir}/model.zip")

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

    # 2 qubits, 10 gates

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 2 qubits, 15 gates

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 2 qubits, 20 gates

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 2 qubits, 25 gates

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=2,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 3 qubits, 10 gates

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 3 qubits, 15 gates

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 3 qubits, 20 gates

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 3 qubits, 25 gates

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=3,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 4 qubits, 10 gates

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=10,
        max_steps=10,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 4 qubits, 15 gates

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=15,
        max_steps=15,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 4 qubits, 20 gates

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=20,
        max_steps=20,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # 4 qubits, 25 gates

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=1,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=2,
        tag="base",
        log_dir="logs"
    )

    model, eval_env = run_experiment(
        qubit_num=4,
        gate_count=25,
        max_steps=25,
        targeted_pool_size=10_000,
        max_generation_tries=10_000_000,
        total_timesteps=10_000_000,
        eval_freq=50_000,
        seed=3,
        tag="base",
        log_dir="logs"
    )

    # test_model(model, eval_env, max_steps=10)
