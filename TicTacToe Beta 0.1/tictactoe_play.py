# -*- coding: utf-8 -*-
"""
三子棋(井字棋) AI —— 对弈脚本(推理)
- 加载训练好的 ttt.npz
- 你执 X(先手), AI 执 O
- 输入 1-9 对应棋盘位置落子
- AI 用 Q 表选择最优落子
"""
import numpy as np

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def encode(board):
    code = 0
    for v in board:
        code = code * 3 + v
    return code

def flip(board):
    return [0 if v == 0 else (2 if v == 1 else 1) for v in board]

def winner(board):
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    if all(v != 0 for v in board):
        return 0
    return None

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def draw_board(board):
    """画 ASCII 棋盘(用户视角): 0空 . 1=X 2=O"""
    sym = {0: ' ', 1: 'X', 2: 'O'}
    rows = []
    for r in range(3):
        cells = [sym[board[r*3+c]] for c in range(3)]
        rows.append(f" {cells[0]} | {cells[1]} | {cells[2]} ")
    return "\n---+---+---\n".join(rows) + "\n"

def ai_move(board, Q):
    """AI 视角: 翻转成 AI 视角(把 O 看成自己的1, X看成对手2), 选最大Q"""
    aview = flip(board)
    s_code = encode(aview)
    moves = legal_moves(aview)
    qvals = [Q[s_code, m] for m in moves]
    best = max(qvals)
    best_moves = [m for m, v in zip(moves, qvals) if abs(v - best) < 1e-12]
    return np.random.choice(best_moves)

def main():
    try:
        d = np.load('ttt.npz')
        Q = d['Q']
        ep = int(d['episodes'])
    except Exception as e:
        print(f"无法加载 ttt.npz: {e}")
        print("请先运行训练: python tictactoe_train.py 50000")
        return
    print("=" * 40)
    print(f"三子棋 AI  你执 X(先手)  已训练 {ep} 局")
    print("落子位置编号:  1|2|3  4|5|6  7|8|9")
    print("=" * 40)
    board = [0] * 9
    while True:
        # 用户落子
        print(draw_board(board))
        while True:
            try:
                pos = int(input("轮到你(1-9)或 0 退出: ").strip())
            except ValueError:
                print("请输入数字 1-9"); continue
            if pos == 0:
                print("已退出"); return
            if 1 <= pos <= 9 and board[pos-1] == 0:
                break
            print("该位置已有子或无效, 重新输入")
        board[pos-1] = 1
        w = winner(board)
        if w is not None:
            print(draw_board(board))
            if w == 1: print("🎉 你赢了!")
            elif w == 2: print("AI 赢了... 再来一局?")
            else: print("平局!")
            return
        # AI 落子
        a = ai_move(board, Q)
        board[a] = 2
        w = winner(board)
        if w is not None:
            print(draw_board(board))
            if w == 2: print("🤖 AI 赢了! 你还是打不过它~")
            elif w == 1: print("🎉 你赢了!")
            else: print("平局!")
            return
        print("AI 落子完成, 轮到你:")

if __name__ == '__main__':
    main()
