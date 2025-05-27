import sys
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiSwitchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, num_switches=3):
        super().__init__()
        self.num_switches = num_switches

        # 状态空间和动作空间都是 MultiDiscrete: 每个开关两个状态 0/1
        self.observation_space = spaces.MultiDiscrete([2] * num_switches)
        self.action_space = spaces.MultiDiscrete([2] * num_switches)  # 每个开关切或不切

        self.target_state = np.random.randint(0, 2, size=num_switches)
        self.state = None
        self.render_mode = render_mode
        self.steps = 0
        self.max_steps = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(0, 2, size=self.num_switches)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        self.steps += 1

        # 应用动作: 切换值（异或操作）
        self.state = (self.state + action) % 2

        done = np.array_equal(self.state, self.target_state)
        reward = 1.0 if done else -0.1
        truncated = self.steps >= self.max_steps

        return self.state.copy(), reward, done, truncated, {}

    def render(self):
        output = f"\rCurrent State: {self.state} | Target: {self.target_state}"
        sys.stdout.write(output)
        sys.stdout.flush()
        time.sleep(0.3)

    def close(self):
        pass


if __name__ == "__main__":
    env = MultiSwitchEnv(render_mode="human", num_switches=3)
    q_table = np.zeros([2] * 3 + [2] * 3)  # Q-table shape: (2,2,2,2,2,2)

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.2
    episodes = 100

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        print(f"Episode {episode + 1}")

        while not done:
            obs_idx = tuple(obs)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.unravel_index(
                    np.argmax(q_table[obs_idx]), q_table[obs_idx].shape
                )

            next_obs, reward, done, truncated, _ = env.step(np.array(action))
            next_obs_idx = tuple(next_obs)

            # Q-learning 更新
            best_next = np.max(q_table[next_obs_idx])
            q_table[obs_idx + tuple(action)] += alpha * (
                reward + gamma * best_next - q_table[obs_idx + tuple(action)]
            )

            obs = next_obs
            total_reward += reward

            if env.render_mode == "human":
                env.render()

        print(f"\nTotal Reward: {total_reward}\n")

    env.close()
