import time

import gymnasium as gym
import numpy as np

# 创建环境
env = gym.make(
    "FrozenLake-v1", render_mode="human", is_slippery=False
)  # 非随机策略便于学习

# 初始化Q表
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# 学习参数
alpha = 0.8  # 学习率
gamma = 0.95  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# 训练Q-learning
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # 选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 更新Q表
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

    # 衰减探索率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# 评估并可视化策略
print("\n训练完成，开始可视化测试...\n")
time.sleep(2)

for _ in range(5):  # 可视化5次
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.5)  # 控制速度
    print("回合总奖励：", total_reward)
    time.sleep(1)

env.close()
