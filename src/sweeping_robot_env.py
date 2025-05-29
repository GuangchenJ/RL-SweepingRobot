import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SweepingRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.window_size = 512

        # 定义状态空间，在这里状态空间等同于观测空间
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(size), spaces.Discrete(size))
        )
        # 定义动作空间
        self.action_space = spaces.Discrete(4)

        # 设置 agent 的默认位置
        self._agent_default_location = np.array([0, 1])
        # 记录机器人的位置，在后续的 reset 方法中会被初始化，并在 step 方法中更新
        # 此处仅仅是一个占位符，或者是冷启动状态
        self._agent_location = np.array([0, 0])
        # 垃圾、充电桩和障碍物的位置
        self._trash_location = np.array([3, 4])
        self._charging_station_location = np.array([0, 0])
        self._obstacle_location = np.array([2, 2])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.icons = {}

        def load_icon(name, file):
            img = pygame.image.load(file)
            img = pygame.transform.scale(
                img,
                (int(self.window_size / self.size), int(self.window_size / self.size)),
            )
            self.icons[name] = img

        load_icon("robot", "icons/robot.png")
        load_icon("trash", "icons/trash.png")
        load_icon("charger", "icons/charger.png")
        load_icon("obstacle", "icons/block.png")

    def _get_obs(self):
        """返回当前 agent 的位置

        Returns:
            tuple: 当前 agent 的位置
        """
        return tuple(self._agent_location)

    def _get_info(self):
        """返回当前 agent 的位置与垃圾和充电桩的欧几里得距离信息

        Returns:
            list: 当前 agent 的位置与垃圾和充电桩的欧几里得距离信息
        """
        return {
            "distance_to_trash": np.sum(
                np.abs(self._agent_location - self._trash_location)
            ),
            "distance_to_charger": np.sum(
                np.abs(self._agent_location - self._charging_station_location)
            ),
        }

    def reset(self, init_pos=None, seed=None, options=None):
        # 处理 Gymnasium 内部的种子等
        super().reset(seed=seed)

        # 1. 确定所有可能的有效初始位置
        initial_positions = []
        for r in range(self.size):
            for c in range(self.size):
                pos = np.array([r, c])
                if (
                    not np.array_equal(pos, self._obstacle_location)
                    and not np.array_equal(pos, self._trash_location)
                    and not np.array_equal(pos, self._charging_station_location)
                ):
                    initial_positions.append(pos)

        # 2. 从有效位置中随机选择一个
        if not initial_positions:  # 如果没有有效位置（例如网格太小或特殊点太多）
            # 设置一个备用/默认的初始位置，确保它不是充电桩或垃圾
            # (这里的逻辑可以根据具体需求调整，例如抛出错误或选择一个尽可能安全的位置)
            if (
                not np.array_equal(
                    self._agent_default_location, self._charging_station_location
                )
                and not np.array_equal(
                    self._agent_default_location, self._trash_location
                )
                and not np.array_equal(
                    self._agent_default_location, self._obstacle_location
                )
            ):
                self._agent_location = self._agent_default_location
        else:  # 如果有有效位置，那么就从有效位置中随机选择一个初始位置
            # self.np_random 是由 super().reset(seed=seed) 设置的随机数生成器
            # 它保证了如果设置了种子，随机选择是可复现的
            idx = self.np_random.integers(0, len(initial_positions))
            self._agent_location = initial_positions[idx]

        observation = self._get_obs()
        info = self._get_info()

        # 对于 RecordVideo，不需要在 reset 时调用 _render_frame
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            pass

        return observation, info

    def step(self, action):
        reward = 0.0  # 默认奖励，避免未赋值情况

        direction_vectors = {
            0: np.array([-1, 0]),  # 上
            1: np.array([1, 0]),  # 下
            2: np.array([0, -1]),  # 左
            3: np.array([0, 1]),  # 右
        }
        previous_location = np.copy(self._agent_location)
        self._agent_location = self._agent_location + direction_vectors[action]
        # 限定元素在 [0, size-1] 范围内
        self._agent_location = np.clip(self._agent_location, 0, self.size - 1)

        if np.array_equal(self._agent_location, self._obstacle_location):
            self._agent_location = previous_location

        # 定义终止符号
        terminated = False
        if np.array_equal(self._agent_location, self._trash_location):
            reward = 5.0
            terminated = True
        elif np.array_equal(self._agent_location, self._charging_station_location):
            reward = 1.0
            terminated = True

        # 设置截断符号
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        # 对于 RecordVideo，不需要在 step 时主动调用 _render_frame
        # RecordVideo 包装器会在需要时调用 env.render()
        if self.render_mode == "human":
            self._render_frame()

        self._last_action = action
        self._last_reward = reward

        return observation, reward, terminated, truncated, info

    def render(self):
        # render 方法现在必须能处理 'rgb_array' 模式并返回图像
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()  #  对于 human 模式，只渲染不返回
            return None  # Human mode render typically doesn't return
        else:
            super().render()  # 或者 gym.Env.render(self)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size + 300, self.window_size)
            )
            pygame.display.set_caption("Sweeping Robot Env")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if pygame.display.get_init() == 0:
            pygame.init()

        # 初始化字体
        if not hasattr(self, "_info_font"):
            pygame.font.init()
            self._info_font = pygame.font.SysFont("Arial", 20)

        # 创建 canvas
        canvas = pygame.Surface((self.window_size + 300, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # 绘制元素图标
        def draw_icon(name, pos):
            if name in self.icons:
                icon = self.icons[name]
                canvas.blit(
                    icon,
                    (
                        pos[1] * pix_square_size,
                        (self.size - 1 - pos[0]) * pix_square_size,
                    ),
                )

        draw_icon("trash", self._trash_location)
        draw_icon("charger", self._charging_station_location)
        draw_icon("obstacle", self._obstacle_location)
        draw_icon("robot", self._agent_location)

        # 绘制网格
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (128, 128, 128),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (128, 128, 128),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # 绘制右侧信息区域背景
        pygame.draw.rect(
            canvas,
            (240, 240, 240),
            pygame.Rect(self.window_size, 0, 300, self.window_size),
        )

        # 显示 agent 状态信息
        info_lines = []
        if hasattr(self, "_last_action"):
            action_name = ["UP", "DOWN", "LEFT", "RIGHT"][self._last_action]
            info_lines.append(f"Action: {action_name}")
        if hasattr(self, "_last_reward"):
            info_lines.append(f"Reward: {self._last_reward:.1f}")
        info_lines.append(
            f"Pos: ({int(self._agent_location[0] + 1)}, {int(self._agent_location[1] + 1)})"
        )

        for i, line in enumerate(info_lines):
            text_surf = self._info_font.render(line, True, (0, 0, 0))
            canvas.blit(text_surf, (self.window_size + 10, 20 + i * 30))

        # 渲染输出
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            # pygame.quit() # pygame.quit()会卸载所有pygame模块，如果其他地方还需要 pygame，可能会出问题
            self.window = None
        # 确保在所有pygame操作完成后才调用pygame.quit()
        # 对于 RecordVideo, 只要 display 模块关闭即可，pygame.quit()可以在程序完全结束时调用
        if pygame.get_init():  # 检查 pygame 是否已初始化
            pygame.quit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical

    # Policy Network 定义
    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

            # 初始化权重
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            # 添加数值稳定性
            x = torch.clamp(x, min=-10, max=10)  # 防止过大的logits
            return F.softmax(x, dim=-1)

    def state_to_tensor(state, size):
        """将状态转换为one-hot编码的张量"""
        row, col = state
        state_vector = torch.zeros(size * size)
        state_vector[row * size + col] = 1
        return state_vector

    def compute_returns(rewards, gamma):
        """计算折扣回报"""
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + gamma * running_return
            returns.insert(0, running_return)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_ema(data, alpha=0.9):
        """计算指数移动平均(EMA)"""
        ema = []
        if len(data) > 0:
            ema.append(data[0])
            for i in range(1, len(data)):
                ema.append(alpha * ema[-1] + (1 - alpha) * data[i])
        return ema

    def plot_training_results(
        episode_rewards, episode_lengths, save_path="training_results.png"
    ):
        """绘制训练结果图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 计算EMA
        ema_rewards = compute_ema(episode_rewards, alpha=0.99)
        ema_lengths = compute_ema(episode_lengths, alpha=0.99)

        # 绘制奖励曲线
        episodes = list(range(len(episode_rewards)))
        ax1.plot(episodes, episode_rewards, "b-", alpha=0.3, label="Reward")
        ax1.plot(episodes, ema_rewards, "r-", linewidth=2, label="EMA Reward (α=0.99)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("The change of reward")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 绘制episode长度曲线
        ax2.plot(episodes, episode_lengths, "g-", alpha=0.3, label="Step")
        ax2.plot(
            episodes, ema_lengths, "orange", linewidth=2, label="EMA Step (α=0.99)"
        )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Step")
        ax2.set_title("Episode Length")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        # 打印统计信息
        print("\n训练统计信息:")
        print(f"最高奖励: {max(episode_rewards):.2f}")
        print(f"最后100个episode平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"最后100个episode平均步数: {np.mean(episode_lengths[-100:]):.2f}")

    def visualize_policy(env, policy_net, save_path="policy_visualization.png"):
        """可视化学习到的策略"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # 动作箭头的方向
        action_arrows = {
            0: (0, 0.4),  # 上
            1: (0, -0.4),  # 下
            2: (-0.4, 0),  # 左
            3: (0.4, 0),  # 右
        }

        # 绘制网格
        for i in range(env.size + 1):
            ax.axhline(y=i, color="gray", linewidth=0.5)
            ax.axvline(x=i, color="gray", linewidth=0.5)

        # 为每个状态绘制最优动作
        for row in range(env.size):
            for col in range(env.size):
                pos = np.array([row, col])

                # 特殊位置标记
                if np.array_equal(pos, env._trash_location):
                    ax.text(
                        col + 0.5,
                        env.size - row - 0.5,
                        "🗑️",
                        fontsize=30,
                        ha="center",
                        va="center",
                    )
                    continue
                elif np.array_equal(pos, env._charging_station_location):
                    ax.text(
                        col + 0.5,
                        env.size - row - 0.5,
                        "🔌",
                        fontsize=30,
                        ha="center",
                        va="center",
                    )
                    continue
                elif np.array_equal(pos, env._obstacle_location):
                    ax.add_patch(
                        plt.Rectangle(
                            (col, env.size - row - 1), 1, 1, facecolor="gray", alpha=0.5
                        )
                    )
                    ax.text(
                        col + 0.5,
                        env.size - row - 0.5,
                        "🚫",
                        fontsize=20,
                        ha="center",
                        va="center",
                    )
                    continue

                # 获取该状态的动作概率
                state_tensor = state_to_tensor((row, col), env.size).unsqueeze(0)
                with torch.no_grad():
                    action_probs = policy_net(state_tensor).squeeze().numpy()

                # 绘制动作箭头（根据概率调整透明度）
                for action, prob in enumerate(action_probs):
                    if prob > 0.1:  # 只显示概率大于0.1的动作
                        dx, dy = action_arrows[action]
                        ax.arrow(
                            col + 0.5,
                            env.size - row - 0.5,
                            dx * prob,
                            dy * prob,
                            head_width=0.1,
                            head_length=0.1,
                            fc="blue",
                            alpha=prob,
                        )

        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect("equal")
        ax.set_title("学习到的策略可视化\n(箭头方向表示动作，透明度表示概率)")
        ax.set_xlabel("列")
        ax.set_ylabel("行")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def train_policy_gradient(env, policy_net, optimizer, episodes, gamma=0.99):
        """使用REINFORCE算法训练策略网络"""
        episode_rewards = []
        episode_lengths = []

        print("环境信息:")
        print(f"- 垃圾位置: {env._trash_location}")
        print(f"- 充电站位置: {env._charging_station_location}")
        print(f"- 障碍物位置: {env._obstacle_location}\n")

        for episode in range(episodes):
            # 收集一个episode的数据
            states = []
            actions = []
            rewards = []

            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < 100:  # 添加最大步数限制
                # 将状态转换为张量
                state_tensor = state_to_tensor(obs, env.size).unsqueeze(0)

                # 获取动作概率分布
                action_probs = policy_net(state_tensor)

                # 采样动作
                m = Categorical(action_probs)
                action = m.sample()

                # 执行动作
                next_obs, reward, terminated, truncated, _ = env.step(action.item())

                # 记录数据
                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)

                obs = next_obs
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # 计算回报
            returns = compute_returns(rewards, gamma)

            # 标准化回报（只有当有多个步骤时才标准化）
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                # 如果只有一步，不进行标准化
                returns = returns

            # 计算损失并更新网络
            policy_loss = []
            entropy_bonus = 0
            for state, action, G in zip(states, actions, returns):
                action_probs = policy_net(state)
                m = Categorical(action_probs)
                policy_loss.append(-m.log_prob(action) * G)
                # 添加熵正则化，鼓励探索
                entropy_bonus += m.entropy() * 0.01

            # 更新策略网络
            optimizer.zero_grad()
            loss = (
                torch.stack(policy_loss).sum() - entropy_bonus
            )  # 减去熵奖励（鼓励探索）

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"警告：Episode {episode} 出现NaN损失，跳过更新")
                continue

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

            optimizer.step()

            # 检查网络参数是否包含NaN
            for param in policy_net.parameters():
                if torch.isnan(param).any():
                    print(f"错误：Episode {episode} 后网络参数包含NaN，停止训练")
                    return episode_rewards, episode_lengths

            # 打印进度
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(
                    f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}, "
                    f"Average Length: {avg_length:.1f}, Current Reward: {total_reward}"
                )

        return episode_rewards, episode_lengths

    # 创建环境
    env = SweepingRobotEnv(render_mode="rgb_array", size=5)

    # 创建策略网络
    input_size = env.size * env.size  # one-hot编码的状态大小
    hidden_size = 128
    output_size = env.action_space.n

    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)  # 降低学习率

    # 训练参数
    episodes = 20000
    gamma = 0.99

    print("开始训练Policy Gradient...")
    episode_rewards, episode_lengths = train_policy_gradient(
        env, policy_net, optimizer, episodes, gamma
    )

    # 绘制训练结果
    print("\n绘制训练结果...")
    plot_training_results(episode_rewards, episode_lengths)

    # 可视化学习到的策略
    print("\n可视化学习到的策略...")
    visualize_policy(env, policy_net)

    # 测试训练好的策略
    print("\n测试训练好的策略...")
    test_episodes = 10
    test_rewards = []

    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 50:  # 限制最大步数防止死循环
            state_tensor = state_to_tensor(obs, env.size).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        test_rewards.append(total_reward)
        print(f"测试Episode {i + 1}: 总奖励 = {total_reward}, 步数 = {steps}")

    print(f"\n测试平均奖励: {np.mean(test_rewards):.2f}")

    # 保存模型
    torch.save(policy_net.state_dict(), "policy_gradient_model.pth")
    print("模型已保存为 'policy_gradient_model.pth'")

    env.close()
