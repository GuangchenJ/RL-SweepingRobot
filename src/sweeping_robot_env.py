import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SweepingRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

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

        # 遇到墙不能走，用之前的位置代替
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.close()

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
            # pygame.quit() # pygame.quit() 会卸载所有 pygame 模块，如果其他地方还需要 pygame，可能会出问题
            self.window = None
        # 确保在所有 pygame 操作完成后才调用 pygame.quit()
        # 对于 RecordVideo, 只要 display 模块关闭即可，pygame.quit()可以在程序完全结束时调用
        if pygame.get_init():  # 检查 pygame 是否已初始化
            pygame.quit()


if __name__ == "__main__":
    pass
