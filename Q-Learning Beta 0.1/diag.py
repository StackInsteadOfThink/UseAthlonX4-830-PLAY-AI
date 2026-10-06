# -*- coding: utf-8 -*-
import random
from collections import deque
from maze import generate_maze, can_move
from qlearning_train import MazeEnv

env = MazeEnv(5)

def bfs():
    q = deque([env.start])
    seen = {env.start}
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == env.goal:
            return True
        for a in range(4):
            ok, nxt = can_move(env.maze, cx, cy, a)
            if ok and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False

print("path exists (BFS):", bfs())
print("start:", env.start, "goal:", env.goal)
print("maze size:", len(env.maze))

state = env.start
steps = 0
for _ in range(5000):
    a = random.randint(0, 3)
    state, r, done = env.step(a)
    steps += 1
    if done:
        print("random reached goal in", steps, "steps")
        break
else:
    print("random did NOT reach goal in 5000 steps, final pos:", state)
