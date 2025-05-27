我们创建的扫地机器人环境要求如下：

- 一共有 5\*5 = 25 个格子，代表观测空间为 5\*5 的离散网格
- 机器人（Agent）：可以在在网格中移动，是我们主要控制的对象。
- 垃圾 (Trash)：位于 [5, 4] (假设索引从 1 开始，代码中我们会用 0 索引，即 [4, 3])。机器人到达此处捡到垃圾，获得 $+5$ 奖励，回合结束。
- 充电桩 (Charging Station)：位于 [1, 1] (代码中为 [0, 0])。机器人到达此处充电，获得 $+1$ 奖励，回合结束（机器人停止行动）。
- 障碍物 (Obstacle)：位于 [3, 3] (代码中为 [2, 2])。机器人无法进入此格子。

 我们可以使用 
 ```python
spaces.Discrete(
    n: int | np.integer[Any],
    seed: int | np.random.Generator | None = None,
    start: int | np.integer[Any] = 0,
)
 ```
 来构建有限个元素的组成空间，例如：
 ```python
 from gymnasium.spaces import Discrete
 observation_space = Discrete(2, seed=42) # {0, 1}
 observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}
 ```
 构造一个 $\left\{ start, start+1, \dots, start+n-1 \right\}$ 的有限元素的集合，其中内容是离散的。

 在本例中，观测空间是 5\*5 的二维离散格子，动作空间是 上下左右 四个动作，都是离散的情况，这里我们都会采用 [spaces.Discrete()](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete) 方法来构建状态空间、观测空间（在本例中状态空间=观测空间）和动作空间:
```python
# 定义状态空间，在这里状态空间等同于观测空间
self.observation_space = spaces.Tuple(
    (spaces.Discrete(size), spaces.Discrete(size))
)
# 定义动作空间
self.action_space = spaces.Discrete(4)
```
其中，[spaces.Tuple](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple) 将两个有 5 个元素 $\left\{ 0, 1, 2, 3, 4 \right\}$ 的空间做笛卡尔积，得到最终 5\*5 的格子状态空间。动作空间就是简单的上下左右 $\left\{ \texttt{top}, \texttt{down}, \texttt{left}, \texttt{right} \right\}$ 四个离散的动作，分别在这里用 $\left\{ 0, 1, 2, 3 \right\}$ 代替。

 对于其他的空间，我们可以使用 Box、MultiBinary、MultiDiscrete、Text 等，详细请参考：[Fundamental Spaces](https://gymnasium.farama.org/api/spaces/fundamental/#fundamental-spaces)