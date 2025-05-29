import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from sweeping_robot_env import SweepingRobotEnv


# SARSA Q-Table Agent
def state_to_index(state, size):
    return state[0] * size + state[1]


def choose_action(Q, state_idx, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state_idx])


def train_sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.size * env.size, env.action_space.n))
    episode_rewards = []
    episode_lengths = []
    episode_success_flags = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state_idx = state_to_index(obs, env.size)
        action = choose_action(Q, state_idx, epsilon, env.action_space.n)

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state_idx = state_to_index(next_obs, env.size)
            next_action = choose_action(Q, next_state_idx, epsilon, env.action_space.n)

            Q[state_idx, action] += alpha * (
                reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action]
            )

            state_idx = next_state_idx
            action = next_action
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_success_flags.append(1 if total_reward >= 5 else 0)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = np.mean(episode_success_flags[-100:]) * 100
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%"
            )

    return Q, episode_rewards, episode_lengths, episode_success_flags


def visualize_policy_from_q(env, Q, save_path="sarsa_policy.png"):
    _, ax = plt.subplots(figsize=(10, 10))
    action_arrows = {
        0: (0, 0.4),
        1: (0, -0.4),
        2: (-0.4, 0),
        3: (0.4, 0),
    }

    # 载入图标
    trash_img = mpimg.imread("icons/trash.png")
    charger_img = mpimg.imread("icons/charger.png")
    block_img = mpimg.imread("icons/block.png")

    for i in range(env.size + 1):
        ax.axhline(y=i, color="gray", linewidth=0.5)
        ax.axvline(x=i, color="gray", linewidth=0.5)

    for row in range(env.size):
        for col in range(env.size):
            pos = np.array([row, col])
            center_x = col + 0.5
            center_y = env.size - row - 0.5

            if np.array_equal(pos, env._trash_location):
                ax.imshow(
                    trash_img, extent=(col, col + 1, env.size - row - 1, env.size - row)
                )
                continue
            elif np.array_equal(pos, env._charging_station_location):
                ax.imshow(
                    charger_img,
                    extent=(col, col + 1, env.size - row - 1, env.size - row),
                )
                continue
            elif np.array_equal(pos, env._obstacle_location):
                ax.imshow(
                    block_img, extent=(col, col + 1, env.size - row - 1, env.size - row)
                )
                continue

            idx = state_to_index((row, col), env.size)
            best_action = np.argmax(Q[idx])
            dx, dy = action_arrows[best_action]
            ax.arrow(
                center_x,
                center_y,
                dx * 0.5,
                dy * 0.5,
                head_width=0.1,
                head_length=0.1,
                fc="blue",
            )

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect("equal")
    ax.set_title("SARSA Learned Policy")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


def plot_training_results(
    episode_rewards, episode_lengths, episode_success_flags, save_path
):
    episodes = list(range(len(episode_rewards)))

    def ema(data, alpha):
        result = [data[0]]
        for val in data[1:]:
            result.append(alpha * result[-1] + (1 - alpha) * val)
        return result

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    axs[0].plot(episodes, episode_rewards, label="Reward", alpha=0.3)
    axs[0].plot(episodes, ema(episode_rewards, 0.99), label="EMA(0.99)", color="red")
    axs[0].set_title("Episode Reward")
    axs[1].plot(episodes, episode_lengths, label="Length", alpha=0.3)
    axs[1].plot(episodes, ema(episode_lengths, 0.99), label="EMA(0.99)", color="orange")
    axs[1].set_title("Episode Length")
    success_rate = [
        np.mean(episode_success_flags[max(0, i - 99) : i + 1]) * 100 for i in episodes
    ]
    axs[2].plot(episodes, success_rate, label="Success Rate (100)", alpha=0.3)
    axs[2].plot(episodes, ema(success_rate, 0.95), label="EMA(0.95)", color="magenta")
    axs[2].set_title("Success Rate")

    for ax in axs:
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


if __name__ == "__main__":
    os.makedirs("res/sweep_robot/sarsa", exist_ok=True)

    env = SweepingRobotEnv(render_mode=None, size=5)
    Q, rewards, lengths, flags = train_sarsa(env, episodes=10000)

    plot_training_results(
        rewards, lengths, flags, "res/sweep_robot/sarsa/training_results.png"
    )
    visualize_policy_from_q(env, Q, "res/sweep_robot/sarsa/policy_visualization.png")
