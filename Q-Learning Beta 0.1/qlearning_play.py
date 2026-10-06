# -*- coding: utf-8 -*-
"""
推理脚本 —— 拖入迷宫 Q 文件 + 权重 npz，打印 ASCII 迷宫 + AI 走的路径。

用法:
  python qlearning_play.py [迷宫Q文件.q] [权重.npz]
或直接把两个文件拖进命令行窗口。

两个文件是分开的:
  迷宫 Q 文件 (.q)  = 迷宫地图(0通路 1墙)     —— 由 maze_gen.py / qlearning_train.py 生成
  权重 (.npz)       = 训练好的模型(Q 表)     —— 由 qlearning_train.py 训练产出

会输出两张图:
  1. 迷宫地图:  起点 S / 终点 E / 墙 ██
  2. AI 路径图: AI 当前位置 O (小球) / 走过的路径 · / 墙 ██

作者: 速龙训练师 (StackInsteadOfThink)
"""

import sys
import os
import numpy as np
from maze import can_move


def draw_maze(maze, path=None, agent=None):
    """打印 ASCII 迷宫。path: AI 走过的 cell 集合; agent: AI 当前 cell。"""
    size = len(maze)
    pathset = set(path) if path else set()
    for y in range(size):
        line = ""
        for x in range(size):
            if maze[y][x] == 1:
                line += "██"                      # 墙
            elif agent is not None and (x, y) == (agent[0] * 2 + 1,
                                                 agent[1] * 2 + 1):
                line += "O "                      # AI 小球当前位置
            elif (x, y) == (0, 0):
                line += "S "                      # 起点
            elif (x, y) == (size - 1, size - 1):
                line += "E "                      # 终点
            elif (x % 2 == 1 and y % 2 == 1 and
                  ((x - 1) // 2, (y - 1) // 2) in pathset):
                line += "· "                      # AI 走过的路径
            else:
                line += "  "                      # 通路
        print(line)


def main():
    maze_file = sys.argv[1] if len(sys.argv) > 1 else "Q_20x20.q"
    weight_file = sys.argv[2] if len(sys.argv) > 2 else "Q_20x20.npz"

    if not os.path.exists(maze_file):
        print("找不到迷宫 Q 文件 %s，先运行 maze_gen.py 或 qlearning_train.py。" % maze_file)
        sys.exit(1)
    if not os.path.exists(weight_file):
        print("找不到权重 %s，先运行 qlearning_train.py 训练。" % weight_file)
        sys.exit(1)

    d = np.load(maze_file)          # 迷宫 Q 文件 → 地形
    maze = d["maze"]
    n = int(d["n"])
    w = np.load(weight_file)        # 权重 npz → 模型 Q 表
    Q = w["Q"]
    episodes = int(w["episodes"])
    start = (0, 0)
    goal = (n - 1, n - 1)

    print("=" * (len(maze) * 2 + 4))
    print("迷宫 Q 文件: %s  权重: %s  迷宫 %dx%d  训练 %d 轮"
          % (os.path.basename(maze_file), os.path.basename(weight_file), n, n, episodes))
    print("=" * (len(maze) * 2 + 4))

    print("\n[1] 迷宫地图  (S=起点  E=终点  ██=墙):")
    draw_maze(maze)

    # 用训练好的 Q 表贪心走迷宫，收集路径
    state = start
    steps = 0
    path = [state]
    while state != goal and steps < n * n * 20:
        si = state[1] * n + state[0]
        action = int(np.argmax(Q[si]))
        ok, nxt = can_move(maze, state[0], state[1], action)
        if not ok:
            break
        state = nxt
        path.append(state)
        steps += 1

    print("\n[2] AI 走过的路径  (O=当前位置  ·=走过  共 %d 步):" % steps)
    draw_maze(maze, path=path, agent=state)

    print("\n结果: ", end="")
    if state == goal:
        print("🎉 AI 成功走出迷宫！共用 %d 步" % steps)
    else:
        print("AI 没走出来（可能训练轮数不够，试试加大训练轮数）")


if __name__ == "__main__":
    main()
