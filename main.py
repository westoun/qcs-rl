
import matplotlib.pyplot as plt
from quasim.gates.utils import create_matrix, create_controlled_matrix
import random
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results, X_TIMESTEPS
import torch
from typing import List, Union

from environment import StateSeparatorEnv
from target_state_generator import TargetStateGenerator
from utils.circuit import decomplexify_vector, create_random_circuit, update_state
from utils.metrics import min_entanglement_entropy

QUBIT_NUM = 2
GATE_TYPE_COUNT = 4  # Clifford gate set
MAX_STEPS = 10

if __name__ == "__main__":

    state_generator = TargetStateGenerator(
        qubit_num=2, gate_count=20, max_pool_size=20_000, train_test_split=0.3)

    LOG_DIR = "logs/"

    random.seed(1)

    env = StateSeparatorEnv(qubit_num=QUBIT_NUM, max_steps=MAX_STEPS,
                            state_generator=state_generator, eval=False)
    env = Monitor(env, LOG_DIR)

    eval_env = StateSeparatorEnv(
        qubit_num=QUBIT_NUM, max_steps=MAX_STEPS, state_generator=state_generator, eval=True)
    eval_env = Monitor(eval_env, f"{LOG_DIR}eval")
    eval_callback = EvalCallback(
        eval_env, log_path=LOG_DIR, n_eval_episodes=1_000, eval_freq=50_000, deterministic=True, render=False
    )

    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=1_000_000, log_interval=1_000,
                progress_bar=True, callback=eval_callback)

    test_count = 5
    for i in range(test_count):
        print(f"\nTest case {i}")

        obs, info = env.reset()
        print(f"Target state: {info['state']}")

        for step in range(MAX_STEPS):
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
