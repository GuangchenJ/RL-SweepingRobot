import sys
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
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
        self.max_steps = 5

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


class TrainingMonitor:
    """训练监控器，记录各种指标用于绘图"""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_mean = []
        self.q_value_max = []
        self.success_rate = deque(maxlen=20)  # 滑动窗口记录最近 20 个 episode 的成功率
        self.epsilon_values = []

    def record_episode(self, total_reward, episode_length, success):
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.success_rate.append(1.0 if success else 0.0)

    def record_q_values(self, q_table):
        self.q_value_mean.append(np.mean(q_table))
        self.q_value_max.append(np.max(q_table))

    def record_epsilon(self, epsilon):
        self.epsilon_values.append(epsilon)

    def compute_ema(self, data, alpha=0.9):
        """计算指数加权移动平均（EMA）"""
        ema = np.zeros_like(data, dtype=float)
        if len(data) > 0:
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * ema[i - 1] + (1 - alpha) * data[i]
        return ema

    def plot_results(self, save_path="res/training_results.png"):
        """绘制训练结果的综合图表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Multi-Switch Environment Training Metrics", fontsize=16)

        # 设置统一的EMA参数
        ema_alpha = 0.9

        # 1. Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(
            self.episode_rewards,
            alpha=0.3,
            color="blue",
            linewidth=1,
            label="Raw Rewards",
        )
        # 添加EMA平滑线
        if len(self.episode_rewards) > 1:
            ema_rewards = self.compute_ema(self.episode_rewards, ema_alpha)
            ax1.plot(ema_rewards, "b-", linewidth=2.5, label=f"EMA (α={ema_alpha})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Rewards Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths, "g-", alpha=0.3, linewidth=1, label="Raw Steps")
        if len(self.episode_lengths) > 1:
            ema_lengths = self.compute_ema(self.episode_lengths, ema_alpha)
            ax2.plot(ema_lengths, "g-", linewidth=2.5, label=f"EMA (α={ema_alpha})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length (Steps to Complete)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Success Rate
        ax3 = axes[0, 2]
        if len(self.success_rate) > 0:
            success_rates = []
            for i in range(len(self.episode_rewards)):
                end_idx = min(i + 1, len(self.success_rate))
                start_idx = max(0, end_idx - 20)
                if end_idx > start_idx:
                    rate = sum(list(self.success_rate)[start_idx:end_idx]) / (
                        end_idx - start_idx
                    )
                    success_rates.append(rate * 100)
            ax3.plot(
                success_rates, "c-", alpha=0.3, linewidth=1, label="20-Episode Window"
            )
            if len(success_rates) > 1:
                ema_success = self.compute_ema(success_rates, ema_alpha)
                ax3.plot(ema_success, "b-", linewidth=2.5, label=f"EMA (α={ema_alpha})")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_title("Success Rate")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])

        # 4. Q-Value Mean
        ax4 = axes[1, 0]
        ax4.plot(self.q_value_mean, "purple", alpha=0.3, linewidth=1, label="Raw Mean")
        if len(self.q_value_mean) > 1:
            ema_q_mean = self.compute_ema(self.q_value_mean, ema_alpha)
            ax4.plot(
                ema_q_mean, color="purple", linewidth=2.5, label=f"EMA (α={ema_alpha})"
            )
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Mean Q-Value")
        ax4.set_title("Average Q-Value Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Q-Value Max
        ax5 = axes[1, 1]
        ax5.plot(self.q_value_max, "orange", alpha=0.3, linewidth=1, label="Raw Max")
        if len(self.q_value_max) > 1:
            ema_q_max = self.compute_ema(self.q_value_max, ema_alpha)
            ax5.plot(
                ema_q_max, color="orange", linewidth=2.5, label=f"EMA (α={ema_alpha})"
            )
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Max Q-Value")
        ax5.set_title("Maximum Q-Value Over Time")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Epsilon Decay
        ax6 = axes[1, 2]
        if self.epsilon_values:
            ax6.plot(
                self.epsilon_values, "red", alpha=0.3, linewidth=1, label="Raw Epsilon"
            )
            if len(self.epsilon_values) > 1:
                ema_epsilon = self.compute_ema(self.epsilon_values, ema_alpha)
                ax6.plot(
                    ema_epsilon, "red", linewidth=2.5, label=f"EMA (α={ema_alpha})"
                )
            ax6.set_xlabel("Episode")
            ax6.set_ylabel("Epsilon")
            ax6.set_title("Exploration Rate (Epsilon) Decay")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_q_heatmap(self, q_table, episode, save_path="q_heatmap.png"):
        """绘制Q表的热力图（仅适用于小规模Q表）"""
        # 将多维Q表展平为2D用于可视化
        flat_shape = (np.prod(q_table.shape[:3]), np.prod(q_table.shape[3:]))
        q_flat = q_table.reshape(flat_shape)

        plt.figure(figsize=(10, 8))
        plt.imshow(q_flat, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Q-Value")
        plt.title(f"Q-Table Heatmap at Episode {episode}")
        plt.xlabel("Action Combinations")
        plt.ylabel("State Combinations")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)

    env = MultiSwitchEnv(render_mode="rgb_array", num_switches=3)
    q_table = np.zeros([2] * 3 + [2] * 3)  # Q-table shape: (2,2,2,2,2,2)

    # 超参数
    alpha = 0.1  # 学习率
    gamma = 0.95  # 折扣因子
    epsilon_start = 0.9  # 初始探索率
    epsilon_end = 0.01  # 最终探索率
    epsilon_decay = 0.995  # 探索率衰减
    episodes = 1000  # 增加训练轮数以更好观察曲线

    # 创建训练监控器
    monitor = TrainingMonitor()

    # 训练循环
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}/{episodes} (ε={epsilon:.3f})")

        while not done and not truncated:
            obs_idx = tuple(obs)

            # ε-贪婪策略
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
            steps += 1

            if env.render_mode == "human":
                env.render()

        # 记录训练数据
        monitor.record_episode(total_reward, steps, done)
        monitor.record_q_values(q_table)
        monitor.record_epsilon(epsilon)

        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(
            f"\nTotal Reward: {total_reward:.2f} | Steps: {steps} | Success: {'Yes' if done else 'No'}"
        )

        # 每 100 个 episode 保存一次Q表热力图
        if (episode + 1) % 100 == 0:
            monitor.plot_q_heatmap(
                q_table, episode + 1, f"res/q_heatmap_episode_{episode + 1}.png"
            )

    env.close()

    # 绘制所有训练结果
    print("\n正在生成训练结果图表...")
    monitor.plot_results("multiswitch_training_results.png")

    # 打印最终统计信息
    print("\n=== 训练完成 ===")
    print(f"最终成功率: {np.mean(list(monitor.success_rate)) * 100:.1f}%")
    print(f"最后 10 轮平均奖励: {np.mean(monitor.episode_rewards[-10:]):.2f}")
    print(f"Q 表平均值: {np.mean(q_table):.4f}")
    print(f"Q 表最大值: {np.max(q_table):.4f}")

    # 测试训练好的策略
    print("\n=== 测试最终策略 (贪婪策略) ===")
    test_episodes = 10
    test_rewards = []

    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            obs_idx = tuple(obs)
            # 使用纯贪婪策略
            action = np.unravel_index(
                np.argmax(q_table[obs_idx]), q_table[obs_idx].shape
            )
            obs, reward, done, truncated, _ = env.step(np.array(action))
            total_reward += reward

        test_rewards.append(total_reward)
        print(
            f"测试 {i + 1}: 奖励 = {total_reward:.2f}, 成功 = {'是' if done else '否'}"
        )

    print(f"\n测试平均奖励: {np.mean(test_rewards):.2f}")
    print(f"测试成功率: {sum(r > 0 for r in test_rewards) / test_episodes * 100:.0f}%")
