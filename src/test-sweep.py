import numpy as np

from sweeping_robot_env import (
    SweepingRobotEnv,  # 假设你的类放在 sweeping_robot_env.py 中
)

env = SweepingRobotEnv(render_mode="human", size=6)
q_table = np.zeros((env.size, env.size, env.action_space.n))

# Q-learning 参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 500

for episode in range(episodes):
    obs, info = env.reset(init_pos=np.array([2, 2]))
    done = False
    total_reward = 0

    while not done:
        row, col = obs
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[row, col])  # 利用

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_row, next_col = next_obs

        # 更新 Q 表
        q_predict = q_table[row, col, action]
        q_target = reward + gamma * np.max(q_table[next_row, next_col])
        q_table[row, col, action] += alpha * (q_target - q_predict)

        obs = next_obs
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
