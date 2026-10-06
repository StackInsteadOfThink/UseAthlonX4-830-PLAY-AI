# -*- coding: utf-8 -*-
"""
迷宫生成模块 —— 纯 Python 实现，不依赖任何第三方库。

用「递归回溯(DFS)」生成完美迷宫：每格都有通路，起点到终点唯一连通，
既有死胡同也有正确路径，适合 Q-Learning 训练。

作者: 速龙训练师 (StackInsteadOfThink)
"""

import random


def generate_maze(n=7, seed=None):
    """
    生成一个 n x n 格子的完美迷宫。

    返回: (2n+1) x (2n+1) 的二维数组 maze，其中
          0 = 通路, 1 = 墙
          起点在 (0,0)，终点在 (2n, 2n)
    """
    rng = random.Random(seed)

    # 初始化全墙
    maze = [[1] * (2 * n + 1) for _ in range(2 * n + 1)]

    # 递归回溯：从 (0,0) cell 出发，随机打通墙
    visited = set([(0, 0)])
    stack = [(0, 0)]

    while stack:
        cx, cy = stack[-1]  # 当前 cell
        # 四个方向的邻居
        neighbors = [
            (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1),
        ]
        # 找出还没访问、且在界内的邻居
        unvisited = [
            (nx, ny) for (nx, ny) in neighbors
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited
        ]

        if unvisited:
            # 随机选一个邻居
            nx, ny = rng.choice(unvisited)
            # 打通 cell 中心与两个 cell 之间的墙
            x1, y1 = cx * 2 + 1, cy * 2 + 1
            x2, y2 = nx * 2 + 1, ny * 2 + 1
            maze[y1][x1] = 0
            maze[(y1 + y2) // 2][(x1 + x2) // 2] = 0
            maze[y2][x2] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    # 起点 / 终点开口
    maze[0][0] = 0
    maze[2 * n][2 * n] = 0

    return maze


def cell_to_pixel(cx, cy):
    """cell 坐标 -> maze 数组坐标"""
    return cx * 2 + 1, cy * 2 + 1


def pixel_to_cell(x, y):
    """maze 数组坐标 -> cell 坐标"""
    return (x - 1) // 2, (y - 1) // 2


def can_move(maze, cx, cy, action):
    """
    判断从 cell (cx,cy) 能否按 action 移动。
    action: 0=上 1=下 2=左 3=右
    返回: (可移动?, 目标cell(nx,ny) 或 None)
    """
    n = (len(maze) - 1) // 2
    dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    dx, dy = dirs[action]
    nx, ny = cx + dx, cy + dy

    if not (0 <= nx < n and 0 <= ny < n):
        return False, None  # 出界

    # 检查两 cell 之间是否有墙
    px, py = cell_to_pixel(cx, cy)
    tx, ty = cell_to_pixel(nx, ny)
    if maze[(py + ty) // 2][(px + tx) // 2] == 0:
        return True, (nx, ny)
    return False, None


def print_maze(maze, path=None):
    """
    把迷宫打印成 ASCII 文本。
    path: 可选，智能体走过的 cell 坐标列表，会用 * 标出来。
    """
    cell_set = set(path) if path else set()
    for y in range(len(maze)):
        line = ""
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                line += "██"  # 墙
            elif (x % 2 == 1 and y % 2 == 1 and
                  pixel_to_cell(x, y) in cell_set):
                line += "◆"  # 走过的路
            else:
                line += "  "  # 通路
        print(line)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    m = generate_maze(n, seed)
    print("迷宫尺寸:", len(m), "x", len(m[0]))
    print("起点:(0,0)  终点:(%d,%d)" % (2 * n, 2 * n))
    print_maze(m)
