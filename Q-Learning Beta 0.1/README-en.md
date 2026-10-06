# Q-Learning Maze Solver

A reinforcement-learning mini project trained on an Athlon X4 830 (pure CPU, no GPU): the AI uses **Q-learning** to trial-and-error through a maze over and over, learning to walk from the entrance to the exit.

## What each file does

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher — just double-click this**, you don't need the other scripts |
| `maze.py` | Maze core code (generate, print, walk logic) |
| `maze_gen.py` | Generates maze terrain, saves it as a `.q` file |
| `qlearning_train.py` | Trains the AI, saves the experience as `.npz` weights |
| `qlearning_play.py` | Runs the trained AI through the maze, prints the path |
| `diag.py` | Debug helper script, usually not needed |
| `Q_20x20.q` / `Q_20x20.npz` | Ready **20×20** maze terrain + trained weights (solves in ~70 steps) |
| `Q_5x5.q` / `Q_5x5.npz` | Ready **5×5** maze terrain + weights (solves in 16 steps) |

## How to use

1. Double-click `menu.bat`.
2. Choose **1 Generate maze** → input a size (e.g. 5 / 12 / 20), generates a new `.q` terrain file.
3. Choose **2 Train** → input size and episodes (e.g. 2000 / 30000 / 1000000), trains the `.npz` weights.
4. Choose **3 Play** → input the maze file name and weight file name, watch the AI solve the maze.

**Don't want to train yourself?** Just choose **3**, type the ready `Q_20x20.q` and `Q_20x20.npz`, and see it in action.

## Notes

- `.q` = maze terrain, `.npz` = AI weights, **don't mix them up**.
- `menu.bat` hard-codes `D:\Q-Learning Beta 0.1` — if you extract elsewhere, edit the `cd /d` line at the top.
- Requires Python 3.12 + numpy.
