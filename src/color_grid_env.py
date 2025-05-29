import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class ColorHexMatchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.rows = 2
        self.cols = 4  # 3 color parts + 1 final result per row
        self.cell_size = 100
        self.window_size = (self.cols * self.cell_size, self.rows * self.cell_size)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([16] * 6)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.target_color = None
        self.agent_color = None

    def reset(self, target_color=np.array([0.5, 0.5, 0.5]), seed=None, options=None):
        super().reset(seed=seed)
        self.target_color = target_color
        self.agent_color = np.zeros_like(self.target_color)
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            frame = self._render_frame()
            return self.target_color, {"frame": frame}
        return self.target_color, {}

    def _hex_to_rgb(self, hex_str):
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return np.array([r, g, b]) / 255.0

    def _digits_to_rgb(self, digits):
        r = digits[0] * 16 + digits[1]
        g = digits[2] * 16 + digits[3]
        b = digits[4] * 16 + digits[5]
        return np.array([r, g, b]) / 255.0

    def step(self, action):
        # 将 action reshape 成 6 个 hex digit
        digits = np.array(action).reshape((6,))
        rgb = self._digits_to_rgb(digits)  # 转换为 RGB 值
        self.agent_color = rgb.astype(np.float32)

        # 计算欧几里得距离作为误差
        error = np.linalg.norm(self.agent_color - self.target_color)
        reward = -error  # 越小越好

        terminated = True
        truncated = False
        info = {
            "agent_rgb": self.agent_color,
            "target_rgb": self.target_color,
            "error": error,
        }

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            info["frame"] = self._render_frame()

        return self.agent_color, reward, terminated, truncated, info

    def _rgb_to_hex(self, rgb):
        return "#{:02X}{:02X}{:02X}".format(*(rgb * 255).astype(int))

    def _render_frame(self):
        pygame.init()
        surface = pygame.Surface(self.window_size)
        font = pygame.font.SysFont("Arial", 14)

        for row, color_set in enumerate([self.target_color, self.agent_color]):
            for col in range(self.cols):
                x = col * self.cell_size
                y = row * self.cell_size

                if col < 3:
                    channel_color = np.zeros(3)
                    channel_color[col] = color_set[col]
                    rect_color = (channel_color * 255).astype(np.uint8)

                    pygame.draw.rect(
                        surface,
                        rect_color,
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                    )
                    pygame.draw.rect(
                        surface,
                        (0, 0, 0),
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                        2,
                    )

                    rgb_text = font.render(f"{color_set[col]:.2f}", True, (0, 0, 0))
                    surface.blit(rgb_text, (x + 5, y + 5))

                elif col == 3:
                    final_color = (color_set * 255).astype(np.uint8)
                    pygame.draw.rect(
                        surface,
                        final_color,
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                    )
                    pygame.draw.rect(
                        surface,
                        (0, 0, 0),
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                        3,
                    )

                    hex_str = self._rgb_to_hex(color_set)
                    hex_text = font.render(hex_str, True, (0, 0, 0))
                    surface.blit(hex_text, (x + 5, y + 40))

        for i in range(1, self.cols):
            pygame.draw.line(
                surface,
                (150, 150, 150),
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size[1]),
                2,
            )

        if self.render_mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("Color Hex Match")
                self.clock = pygame.time.Clock()

            self.window.blit(surface, (0, 0))
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(surface), (1, 0, 2)
            )  # shape: (H, W, 3)

    def close(self):
        if self.window is not None:
            pygame.quit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import trange

    class CEMAgent:
        def __init__(
            self, action_dim=6, action_bins=16, pop_size=100, elite_frac=0.2, sigma=1.0
        ):
            self.action_dim = action_dim
            self.action_bins = action_bins
            self.pop_size = pop_size
            self.elite_frac = elite_frac
            self.n_elite = int(pop_size * elite_frac)

            # 初始化为每个 HEX digit 的均匀 logits（每位16个logits）
            self.logits_mean = np.zeros((action_dim, action_bins))
            self.logits_std = np.ones((action_dim, action_bins)) * sigma

        def sample_actions(self):
            # 从正态分布中采样 logit，然后 softmax 得到概率
            samples = (
                np.random.randn(self.pop_size, self.action_dim, self.action_bins)
                * self.logits_std
                + self.logits_mean
            )
            probs = F.softmax(torch.tensor(samples), dim=2).numpy()
            actions = np.array(
                [
                    np.array(
                        [
                            np.random.choice(self.action_bins, p=probs[i, j])
                            for j in range(self.action_dim)
                        ]
                    )
                    for i in range(self.pop_size)
                ]
            )
            return actions, samples

        def update_distribution(self, elite_logits):
            self.logits_mean = np.mean(elite_logits, axis=0)
            self.logits_std = np.std(elite_logits, axis=0) + 1e-5  # 防止为0

        def act(self):
            # 用当前 logits 概率选择一个动作（用于测试）
            probs = F.softmax(torch.tensor(self.logits_mean), dim=1).numpy()
            return np.array(
                [
                    np.random.choice(self.action_bins, p=probs[i])
                    for i in range(self.action_dim)
                ]
            )

    import time

    target_color = np.random.rand(3).astype(np.float32)
    print(f"target color: {target_color}")

    env = ColorHexMatchEnv(render_mode="human")
    agent = CEMAgent(pop_size=100, elite_frac=0.2)

    n_iters = 200
    reward_history = []

    for iteration in trange(n_iters):
        actions, sampled_logits = agent.sample_actions()
        rewards = []

        for i in range(agent.pop_size):
            obs, _ = env.reset()
            action = actions[i]
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)

        rewards = np.array(rewards)
        elite_idxs = rewards.argsort()[-agent.n_elite :]
        elite_logits = sampled_logits[elite_idxs]
        agent.update_distribution(elite_logits)

        mean_reward = rewards[elite_idxs].mean()
        reward_history.append(mean_reward)

        if iteration % 10 == 0:
            best_action = agent.act()
            _, _ = env.reset(target_color=target_color)
            _, reward, _, _, info = env.step(best_action)
            print(
                f"[{iteration}] Best Hex: {env._rgb_to_hex(env.agent_color)}, Error: {-reward:.4f}"
            )
            time.sleep(0.5)

    env.close()

    plt.plot(reward_history)
    plt.title("CEM Average Elite Reward Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Elite Reward")
    plt.grid(True)
    plt.show()
