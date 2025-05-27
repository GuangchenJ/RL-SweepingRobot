import numpy as np

# 网格世界参数
GRID_SIZE = 5
ACTIONS = ["⬆️", "⬇️", "⬅️", "➡️"]
ACTION_DELTAS = {"⬆️": (-1, 0), "⬇️": (1, 0), "⬅️": (0, -1), "➡️": (0, 1)}
GAMMA = 0.8  # 折扣因子
THETA = 1e-5  # 收敛阈值
REWARD = 0  # 每步移动的奖励

# 目标状态和障碍物
GOAL_STATE = (1, 4)
BATTERY = (4, 0)
OBSTACLE = (2, 2)

# 初始化状态值
V = np.zeros((GRID_SIZE, GRID_SIZE))
V[GOAL_STATE] = 3 / 0.8
V[OBSTACLE] = -10 / 0.8
V[BATTERY] = 1 / 0.8


# 值迭代算法
def value_iteration():
    while True:
        delta = 0
        new_V = np.copy(V)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = (i, j)
                if state == GOAL_STATE or state == OBSTACLE or state == BATTERY:
                    continue

                # 计算每个动作的Q值
                q_values = []
                for action in ACTIONS:
                    di, dj = ACTION_DELTAS[action]
                    next_i, next_j = i + di, j + dj

                    # 检查是否越界或遇到障碍物
                    if (
                        (0 <= next_i < GRID_SIZE)
                        and (0 <= next_j < GRID_SIZE)
                        and (next_i, next_j) != OBSTACLE
                    ):
                        next_state = (next_i, next_j)
                        q_value = REWARD + GAMMA * V[next_state]
                    else:
                        # 如果越界或遇到障碍物，留在原状态
                        q_value = REWARD + GAMMA * V[i, j]
                    q_values.append(q_value)

                # 选择最大的Q值作为新状态值
                new_V[i, j] = max(q_values)
                delta = max(delta, abs(new_V[i, j] - V[i, j]))

        V[:] = new_V
        print(V)
        if delta < THETA:
            break


# 提取最优策略
def extract_policy(V):
    policy = np.full((GRID_SIZE, GRID_SIZE), "", dtype=object)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = (i, j)
            if state == GOAL_STATE:
                policy[i, j] = "🗑️"
                continue
            elif state == OBSTACLE:
                policy[i, j] = "♿️"
                continue
            elif state == BATTERY:
                policy[i, j] = "🔋"
                continue

            # 计算每个动作的Q值，选择最大的
            max_q = float("-inf")
            best_action = None
            for action in ACTIONS:
                di, dj = ACTION_DELTAS[action]
                next_i, next_j = i + di, j + dj
                if (
                    (0 <= next_i < GRID_SIZE)
                    and (0 <= next_j < GRID_SIZE)
                    and (next_i, next_j) != OBSTACLE
                ):
                    q_value = REWARD + GAMMA * V[next_i, next_j]
                else:
                    q_value = REWARD + GAMMA * V[i, j]
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
            policy[i, j] = best_action
    return policy


# 运行值迭代
value_iteration()

# 获取最优策略
policy = extract_policy(V)

V[GOAL_STATE] = 0
V[OBSTACLE] = 0
V[BATTERY] = 0

# 打印最优状态值
print("最优状态值:")
print(np.round(V, 3))

# 打印最优策略
print("\n最优策略:")
for i in range(GRID_SIZE):
    row = []
    for j in range(GRID_SIZE):
        row.append(policy[i, j])
    print(row)
