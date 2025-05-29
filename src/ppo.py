"""
PPO Training & Evaluation – Double Mountain Car
===============================================

* Action space*: **Discrete(9)** (`NineToMultiAction`)
* Reward*: dense shaping via `DenseRewardWrapper`
* Normalisation*: `VecNormalize` for obs & reward

All artefacts are written to **`res/double_mountain_car/`**.

Plots generated (raw curve + EMA α = 0.95):
* `reward_curve.png`   — episode reward
* `episode_length.png` — episode length / steps
* `success_rate.png`   — success rate (if `is_success` info present)

TensorBoard is **disabled** to keep dependencies minimal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------------------------------------------------------
# 0.  Helper wrappers
# -----------------------------------------------------------------------------

COMBOS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)


class NineToMultiAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)

    def action(self, act: int):
        return COMBOS[act]


class DenseRewardWrapper(gym.Wrapper):
    """Dense shaping: add avg Δx × weight per step."""

    def __init__(self, env: gym.Env, weight: float = 2.0):
        super().__init__(env)
        self.weight = weight
        self._last_pos = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_pos = np.array([obs[0], obs[2]])
        return obs, info

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        pos = np.array([obs[0], obs[2]])
        shaped = r + self.weight * (pos - self._last_pos).mean()
        self._last_pos = pos
        return obs, shaped, term, trunc, info


# -----------------------------------------------------------------------------
# 1.  I/O paths
# -----------------------------------------------------------------------------

RES_DIR = Path("res") / "double_mountain_car"
RES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = RES_DIR / "ppo_double_mountain_car.zip"
VECNORM_PATH = RES_DIR / "vecnorm.pkl"

MAX_EPISODE_STEPS = 400

# -----------------------------------------------------------------------------
# 2.  Environment factory
# -----------------------------------------------------------------------------


def make_env() -> gym.Env:
    from double_mountain_car_env import DoubleMountainCarEnv

    base = DoubleMountainCarEnv(render_mode=None)
    base = NineToMultiAction(base)
    base = DenseRewardWrapper(base)
    base = TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    return Monitor(base, str(RES_DIR))


# -----------------------------------------------------------------------------
# 3.  Training
# -----------------------------------------------------------------------------

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=1024,
    batch_size=512,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,
    device="cpu",
)

TOTAL_STEPS = 1_000_000
print(f"Training for {TOTAL_STEPS:,} steps …\n")
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save(MODEL_PATH)
vec_env.save(str(VECNORM_PATH))
print("Training finished. Model & VecNormalize saved.\n")


# -------------------------------------------------------------------
# 4.  Plot helper (raw + EMA α=0.9)
# -------------------------------------------------------------------
def compute_ema(arr, alpha: float = 0.9):
    """EMA, 支持 list / ndarray / Series，返回 ndarray"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * arr[i]
    return ema


def plot_series(y, title: str, ylabel: str, fname: str):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    ema = compute_ema(y, alpha=0.95)
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(x, y, alpha=0.3, label="raw")
    plt.plot(x, ema, label="EMA α=0.95", lw=2)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / fname)
    plt.close()


# -------------------------------------------------------------------
# 5.  Plot metrics from monitor.csv
# -------------------------------------------------------------------
monitor_file = RES_DIR / "monitor.csv"
if monitor_file.exists():
    df = pd.read_csv(monitor_file, skiprows=1)

    if "r" in df:
        plot_series(df["r"], "Episode Reward", "Reward", "reward_curve.png")

    if "l" in df:
        plot_series(df["l"], "Episode Length", "Steps", "episode_length.png")

        # 成功率：步数 < TimeLimit → 1，否则 0
        success = (df["l"] < MAX_EPISODE_STEPS).astype(int)
        plot_series(
            success.rolling(50).mean().fillna(0),
            "Success Rate (50-ep MA)",
            "Rate",
            "success_rate.png",
        )

# -----------------------------------------------------------------------------
# 6.  Evaluation only block (optional)
# -----------------------------------------------------------------------------


def evaluate(trials: int = 20):
    raw_eval = DummyVecEnv([make_env])
    eval_env = VecNormalize.load(str(VECNORM_PATH), raw_eval)
    model_ = PPO.load(str(MODEL_PATH), env=eval_env, device="cpu")

    mean_r, std_r = evaluate_policy(
        model_,
        eval_env,
        n_eval_episodes=trials,
        deterministic=True,
    )
    print(f"Mean reward over {trials} episodes: {mean_r:.2f} ± {std_r:.2f}")


if __name__ == "__main__":
    evaluate()
