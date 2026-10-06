# -*- coding: utf-8 -*-
"""
三子棋(井字棋) AI —— self-play Q-Learning 训练脚本
- 状态: 棋盘9格, 值 0=空 1=当前玩家视角的"我" 2=对手(归一化视角)
- 动作: 9个位置, 合法动作=空格
- Q(s,a): 当前玩家落子a的期望终局回报(视角:+1赢 / -1输 / 0平)
- 训练: AI自己跟自己下(交替视角翻转), 走完一局后按零和规则倒推更新
- 产物: ttt.npz (含Q表 + episodes)
"""
import sys
import numpy as np

# 三子棋获胜线
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def encode(board):
    """把9格(0/1/2)编码成 0..3^9-1 的整数状态编号"""
    code = 0
    for v in board:
        code = code * 3 + v
    return code

def flip(board):
    """翻转视角: 1<->2 (轮到对手时把我的棋子看成对手的)"""
    return [0 if v == 0 else (2 if v == 1 else 1) for v in board]

def winner(board):
    """返回胜者: 1=视角玩家赢, 2=对手赢, 0=平, None=未结束"""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    if all(v != 0 for v in board):
        return 0
    return None

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def train(episodes, alpha=0.1, gamma=0.9, eps_start=1.0, eps_end=0.05, eps_decay=0.9995, verbose=True):
    Q = np.zeros((3 ** 9, 9), dtype=np.float64)
    eps = eps_start
    win = lose = draw = 0
    for ep in range(episodes):
        board = [0] * 9            # 当前玩家视角棋盘
        history = []               # 每手 (状态编码, 动作)
        while True:
            s_code = encode(board)
            moves = legal_moves(board)
            if not moves:
                break
            if np.random.rand() < eps:
                a = np.random.choice(moves)
            else:
                qvals = [Q[s_code, m] for m in moves]
                best = max(qvals)
                best_moves = [m for m, v in zip(moves, qvals) if abs(v - best) < 1e-12]
                a = np.random.choice(best_moves)
            history.append((s_code, a))
            board[a] = 1            # 当前玩家落子
            w = winner(board)       # 当前玩家视角的胜者
            if w is not None:
                break
            board = flip(board)     # 翻转到对手视角
        # 终局回报(从最后落子者视角)
        w = winner(board)
        if w == 1:
            r = 1.0; win += 1
        elif w == 2:
            r = -1.0; lose += 1
        else:
            r = 0.0; draw += 1
        # 倒推更新(零和: 对手视角取反)
        G = r
        for (s_code, a) in reversed(history):
            Q[s_code, a] += alpha * (G - Q[s_code, a])
            G = -gamma * G
        eps = max(eps_end, eps * eps_decay)
        if verbose and (ep + 1) % 2000 == 0:
            print(f"episode {ep+1:>7} | 胜 {win:>5} 负 {lose:>5} 平 {draw:>5} | eps {eps:.3f}")
    np.savez('ttt.npz', Q=Q, episodes=episodes, win=win, lose=lose, draw=draw)
    print("训练完成！权重已保存到: ttt.npz")
    print(f"对局统计: 胜 {win} 负 {lose} 平 {draw} (总 {win+lose+draw})")
    return Q

if __name__ == '__main__':
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    print("=" * 60)
    print(f"三子棋 AI 训练开始  自对弈 {episodes} 局")
    print("=" * 60)
    train(episodes)
