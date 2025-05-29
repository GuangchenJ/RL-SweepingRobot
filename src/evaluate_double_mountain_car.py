# -------------------------------------------------
# evaluate_and_plot_fixed.py
# -------------------------------------------------
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------- 路径 ----------
RESULT_DIR = Path("res/double_mountain_car")
MODEL_PATH = RESULT_DIR / "ppo_double_mountain_car.zip"
VECNORM_PATH = RESULT_DIR / "vecnorm.pkl"
N_EVAL_EPISODES, MAX_STEPS, EMA_ALPHA = 20, 400, 0.9


# ---------- EMA ----------
# ---------- EMA ----------
def compute_ema(data, alpha=0.9):
    """
    data : list 或 1-D ndarray
    返回值与 data 等长的 ndarray
    """
    data_arr = np.asarray(data, dtype=float)
    if data_arr.size == 0:
        return data_arr
    ema = np.empty_like(data_arr)
    ema[0] = data_arr[0]
    for i in range(1, data_arr.size):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * data_arr[i]
    return ema


def plot_curve(series, title, ylabel, fname, ema_alpha=0.9):
    """
    series 可以是 list 或 ndarray
    保存到 RESULT_DIR / fname
    """
    series = np.asarray(series, dtype=float)
    ema = compute_ema(series, alpha=ema_alpha)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(series, label="raw", alpha=0.3)
    plt.plot(ema, label=f"EMA α={ema_alpha}", lw=2)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / fname)
    plt.close()


# ---------- 环境包装器（与训练一致，略） ----------
COMBOS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))


class NineToMultiAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)

    def action(self, a: int):
        return COMBOS[a]


class DenseRewardWrapper(gym.Wrapper):
    def __init__(self, env, w=2.0):
        super().__init__(env)
        self.w = w
        self._last = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last = (obs[0], obs[2])
        return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        dx1, dx2 = obs[0] - self._last[0], obs[2] - self._last[1]
        r += self.w * (dx1 + dx2) / 2
        self._last = (obs[0], obs[2])
        return obs, r, term, trunc, info


def make_env():
    from double_mountain_car_env import DoubleMountainCarEnv

    env = DoubleMountainCarEnv(render_mode=None)
    env = NineToMultiAction(env)
    env = DenseRewardWrapper(env)
    env = TimeLimit(env, MAX_STEPS)
    return Monitor(env)


# ---------- 载入模型 ----------
raw_eval = DummyVecEnv([make_env])
eval_env = VecNormalize.load(str(VECNORM_PATH), raw_eval)
eval_env.training = False
eval_env.norm_reward = False
model = PPO.load(str(MODEL_PATH), env=eval_env, device="cpu")

# ---------- 评估 ----------
ep_r, ep_len, ep_succ = [], [], []
for _ in range(N_EVAL_EPISODES):
    obs = eval_env.reset()  # VecEnv: 只返回 obs
    done = [False]
    tot_r, steps = 0.0, 0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, infos = eval_env.step(action)  # 4 元组
        tot_r += rewards[0]
        steps += 1
    ep_r.append(tot_r)
    ep_len.append(steps)
    # 成功: 回合未触发 TimeLimit → 步数 < MAX_STEPS
    ep_succ.append(1 if steps < MAX_STEPS else 0)

print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: "
    f"{np.mean(ep_r):.2f} ± {np.std(ep_r):.2f}"
)

# ---------- 绘图 ----------
plot_curve(ep_r, "Eval Episode Reward", "Reward", "eval_reward_curve.png")
plot_curve(ep_len, "Eval Episode Length", "Steps", "eval_episode_length.png")
cum_succ = np.cumsum(ep_succ) / (np.arange(len(ep_succ)) + 1)
plot_curve(cum_succ, "Eval Success Rate", "Success Rate", "eval_success_rate.png")
