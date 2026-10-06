# -*- coding: utf-8 -*-
"""
Q-Learning 训练脚本 —— 让 AI 学会走迷宫。

纯 numpy 实现 e-greedy Q-Learning：
  - 状态 = 智能体所在格子 (cx, cy)
  - 动作 = 上 / 下 / 左 / 右
  - 奖励 = 到达终点 +100（终止）、撞墙 -1（留在原地）、每走一步 -0.1
训练完成后把 Q 表 + 迷宫 + 参数打包成 .npz，供演示脚本加载。

用法:  python qlearning_train.py [迷宫尺寸] [训练轮数]
示例:  python qlearning_train.py 7 3000

作者: 速龙训练师 (StackInsteadOfThink)
"""

import sys
import os
import numpy as np
from maze import generate_maze, cell_to_pixel, can_move


# ---------- 环境 ----------
class MazeEnv:
    """一个简单的格子迷宫环境。"""

    def __init__(self, n, seed=None, maze=None):
        self.n = n
        self.maze = maze if maze is not None else generate_maze(n, seed)
        self.start = (0, 0)
        self.goal = (n - 1, n - 1)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        """执行动作，返回 (新状态, 奖励, 是否结束)。"""
        cx, cy = self.state
        ok, nxt = can_move(self.maze, cx, cy, action)

        if not ok:
            # 撞墙 / 出界：留在原地，给个负奖励
            return self.state, -1.0, False

        nx, ny = nxt
        self.state = (nx, ny)             # 关键：推进智能体位置
        if (nx, ny) == self.goal:
            return (nx, ny), 100.0, True  # 到达终点，结束

        # 普通一步：小惩罚，鼓励走最短
        return (nx, ny), -0.1, False


# ---------- Q-Learning ----------
def train(n, episodes, alpha=0.1, gamma=0.9, eps_start=1.0, eps_end=0.05,
          seed=None, eps_decay=0.998, maze=None):
    env = MazeEnv(n, seed, maze)
    num_actions = 4
    num_states = n * n
    Q = np.zeros((num_states, num_actions), dtype=np.float64)

    def state_index(cell):
        cx, cy = cell
        return cy * n + cx

    history = []   # 每轮的步数（能看到 AI 从笨到聪明）
    eps = eps_start
    goal = env.goal

    def phi(cell):
        """势能塑形：离终点越近，值越大（曼哈顿距离的负值）。"""
        cx, cy = cell
        gx, gy = goal
        return -(abs(cx - gx) + abs(cy - gy))

    for ep in range(episodes):
        state = env.reset()
        total_steps = 0
        done = False
        while not done:
            # e-greedy 选动作
            if np.random.rand() < eps:
                action = np.random.randint(0, num_actions)
            else:
                action = int(np.argmax(Q[state_index(state)]))

            next_state, reward, done = env.step(action)
            si = state_index(state)
            nsi = state_index(next_state)

            # 势能奖励塑形：引导 AI 朝终点走
            shaped = reward + gamma * phi(next_state) - phi(state)
            # Q-learning 更新
            best_next = np.max(Q[nsi])
            Q[si][action] += alpha * (
                shaped + gamma * best_next - Q[si][action]
            )

            state = next_state
            total_steps += 1

            # 防止卡死（撞墙也能结束当前轮）
            if total_steps > n * n * 10:
                done = True

        history.append(total_steps)
        # epsilon 衰减
        eps = max(eps_end, eps * eps_decay)

        if (ep + 1) % 500 == 0 or ep == 0:
            avg = np.mean(history[-100:]) if history else 0
            print("episode %5d | 本轮回合步数 %5d | 最近100轮平均 %5.1f | eps %.3f"
                  % (ep + 1, total_steps, avg, eps))

    return env, Q, history


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    weight_path = "Q_%dx%d.npz" % (n, n)   # 权重存 npz（模型）
    qfile = "Q_%dx%d.q" % (n, n)           # 迷宫存 Q 文件

    print("=" * 50)
    print("Q-Learning 训练开始")
    print("迷宫尺寸: %dx%d  训练轮数: %d" % (n, n, episodes))
    print("=" * 50)

    # 若已有迷宫 Q 文件就加载它训练；否则自动生成并存成 Q 文件
    maze = None
    if os.path.exists(qfile):
        d = np.load(qfile)
        maze = d["maze"]
        print("加载迷宫 Q 文件: %s" % qfile)
    else:
        maze = generate_maze(n, None)
        tmp = qfile + ".npz"
        np.savez_compressed(tmp, maze=np.array(maze, dtype=np.int8), n=n, seed=0)
        if os.path.exists(qfile):
            os.remove(qfile)
        os.rename(tmp, qfile)
        print("自动生成迷宫并保存为 Q 文件: %s" % qfile)

    # 大迷宫探索期要更长，否则找不到终点路径
    eps_decay = 0.9999 if n >= 15 else 0.998
    env, Q, history = train(n, episodes, eps_decay=eps_decay, maze=maze)

    # 存权重 npz（只含模型 Q 表 + 参数，迷宫在 Q 文件里）
    np.savez_compressed(
        weight_path,
        Q=Q,
        n=n,
        start=np.array(env.start, dtype=np.int8),
        goal=np.array(env.goal, dtype=np.int8),
        episodes=episodes,
    )
    print("\n训练完成！权重已保存到: %s" % weight_path)
    print("迷宫 Q 文件: %s" % qfile)
    print("最近50轮平均步数: %.1f" % np.mean(history[-50:]))

    # 顺带用训练好的 Q 表走一遍，看步数（贪心）
    greedy_steps = run_greedy(env, Q)
    print("用训练好的 Q 表贪心走迷宫: %d 步到达终点" % greedy_steps)


def run_greedy(env, Q):
    """用贪心策略从起点走到终点，返回步数。"""
    n = env.n
    state = env.reset()
    steps = 0
    while state != env.goal and steps < n * n * 10:
        si = state[1] * n + state[0]
        action = int(np.argmax(Q[si]))
        state, _, done = env.step(action)
        steps += 1
        if done:
            break
    return steps


if __name__ == "__main__":
    main()
