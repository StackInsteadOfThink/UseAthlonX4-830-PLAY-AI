# -*- coding: utf-8 -*-
"""
迷宫生成器 —— 生成迷宫 Q 文件（.q），只含迷宫地形数据。

Q 文件 = 迷宫地图（0=通路 1=墙）。
训练好的权重（Q 表模型）单独存成 .npz，由 qlearning_train.py 生成。

用法:
  python maze_gen.py [尺寸] [随机种子(可选)]
示例:
  python maze_gen.py 20        # 生成 20x20 迷宫 Q 文件
  python maze_gen.py 20 7      # 生成 20x20 迷宫(固定种子,可复现)

产出:  Q_尺寸x尺寸.q   (迷宫文件)

作者: 速龙训练师 (StackInsteadOfThink)
"""

import sys
import os
import numpy as np
from maze import generate_maze


def save_maze_file(n, seed):
    """生成迷宫并保存成 .q 文件，返回 (迷宫数组, 文件名)。"""
    maze = generate_maze(n, seed)
    out = "Q_%dx%d.q" % (n, n)
    tmp = out + ".npz"                     # numpy 会自动补 .npz，先存临时名
    np.savez_compressed(
        tmp,
        maze=np.array(maze, dtype=np.int8),
        n=n,
        seed=seed if seed is not None else 0,
    )
    if os.path.exists(out):
        os.remove(out)
    os.rename(tmp, out)                    # 去掉 .npz 后缀，得到真正的 .q 文件
    return maze, out


def show_preview(maze):
    """ASCII 预览: 起点 S, 终点 E, 墙 ██, 通路空格。"""
    size = len(maze)
    print("=" * (size * 2 + 4))
    for y in range(size):
        line = "|"
        for x in range(size):
            if maze[y][x] == 1:
                line += "██"
            elif (x, y) == (0, 0):
                line += "S "          # 起点
            elif (x, y) == (size - 1, size - 1):
                line += "E "          # 终点
            else:
                line += "  "
        print(line + "|")
    print("=" * (size * 2 + 4))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = None
    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            seed = None

    maze, out = save_maze_file(n, seed)

    print("\n迷宫 Q 文件已生成并保存: %s" % out)
    print("尺寸: %d x %d  起点S(左上)  终点E(右下)  (0=通路 1=墙)" % (n, n))
    show_preview(maze)
    print("提示: 之后用 qlearning_train.py 针对这个迷宫训练，权重会存成 .npz")


if __name__ == "__main__":
    main()
