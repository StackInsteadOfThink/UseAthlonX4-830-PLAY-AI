# -*- coding: utf-8 -*-
"""三子棋 AI 验证: 对随机对手 + 完美Minimax对手, 检验是否真正不败"""
import numpy as np

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def encode(board):
    code=0
    for v in board: code=code*3+v
    return code

def flip(board):
    return [0 if v==0 else (2 if v==1 else 1) for v in board]

def winner(board):
    for a,b,c in WIN_LINES:
        if board[a]==board[b]==board[c] and board[a]!=0: return board[a]
    if all(v!=0 for v in board): return 0
    return None

def legal(board): return [i for i in range(9) if board[i]==0]

def ai_move(board, Q, ai_piece):
    """AI视角: AI的棋子=1, 对手=2. 若AI执X(board里X=1)则直接用board, 若执O(board里O=2)则flip"""
    aview = board[:] if ai_piece == 1 else flip(board)
    sc=encode(aview); moves=legal(aview)
    qv=[Q[sc,m] for m in moves]; best=max(qv)
    bm=[m for m,v in zip(moves,qv) if abs(v-best)<1e-12]
    return np.random.choice(bm)

def minimax(board, player):
    """完美对手. player=当前走棋者(视角固定1先手2后手), 返回(score, move)"""
    # board 用户视角(1=X先手,2=O后手)
    w=winner(board)
    if w==1: return 1, None      # X(先手)赢
    if w==2: return -1, None     # O赢
    if w==0: return 0, None
    moves=legal(board)
    if player==1:  # X 最大化
        best=-2; bm=None
        for m in moves:
            board[m]=1; s,_=minimax(board,2); board[m]=0
            if s>best: best=s; bm=m
        return best,bm
    else:          # O 最小化
        best=2; bm=None
        for m in moves:
            board[m]=2; s,_=minimax(board,1); board[m]=0
            if s<best: best=s; bm=m
        return best,bm

def play(Q, ai_first, opponent, seed=0):
    np.random.seed(seed)
    board=[0]*9
    # ai_first: AI 执 X(1)? 若 ai_first AI是X先手. opponent 执 O后手.
    # 1=X先手, 2=O后手
    turn=1
    while True:
        w=winner(board)
        if w is not None: return w
        if turn==1:  # X 走
            if ai_first: m=ai_move(board,Q,1); board[m]=1
            else: _,m=minimax(board,1); board[m]=1
        else:        # O 走
            if not ai_first: m=ai_move(board,Q,2); board[m]=2
            else: _,m=minimax(board,2); board[m]=2
        turn=3-turn

def random_opponent_play(Q, ai_first, n=200):
    np.random.seed(42)
    res={1:0,2:0,0:0,-1:0}
    for _ in range(n):
        board=[0]*9; turn=1; dead=0
        while True:
            w=winner(board)
            if w is not None: res[w]+=1; break
            if dead>100: res[-1]+=1; break
            if turn==1:
                if ai_first: m=ai_move(board,Q,1); board[m]=1
                else: m=np.random.choice(legal(board)); board[m]=1
            else:
                if not ai_first: m=ai_move(board,Q,2); board[m]=2
                else: m=np.random.choice(legal(board)); board[m]=2
            turn=3-turn; dead+=1
    return res

d=np.load('ttt.npz'); Q=d['Q']; ep=int(d['episodes'])
print(f"权重: ttt.npz (训练{ep}局)  验证开始\n")

# 1) vs 随机
r1=random_opponent_play(Q,True,200)
r2=random_opponent_play(Q,False,200)
print(f"[vs随机] AI先手 200局: 胜{r1[1]} 负{r1[2]} 平{r1[0]}  赢率 {r1[1]/2:.0f}%")
print(f"[vs随机] AI后手 200局: 胜{r2[1]} 负{r2[2]} 平{r2[0]}  赢率 {r2[2]/2:.0f}%(AI是O)")
print()

# 2) vs 完美Minimax
p1=play(Q,True,'mm',1); p2=play(Q,False,'mm',1)
print(f"[vs完美] AI先手: {'AI赢' if p1==1 else ('平局' if p1==0 else 'AI输!')}  (理论:先手应至少不败)")
print(f"[vs完美] AI后手: {'AI赢' if p2==2 else ('平局' if p2==0 else 'AI输!')}  (理论:后手应平局)")
# 多跑几局确认稳定
stats={1:0,2:0,0:0}
for i in range(20):
    s=play(Q,True,'mm',100+i); stats[s]+=1
print(f"[vs完美] AI先手 20局: 胜{stats[1]} 负{stats[2]} 平{stats[0]}")
stats={1:0,2:0,0:0}
for i in range(20):
    s=play(Q,False,'mm',200+i); stats[s]+=1
print(f"[vs完美] AI后手 20局: 胜{stats[1]} 负{stats[2]} 平{stats[0]}  (AI是O, O赢=AI赢)")
