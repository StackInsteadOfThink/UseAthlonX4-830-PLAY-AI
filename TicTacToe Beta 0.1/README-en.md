# TicTacToe AI

A TicTacToe AI trained on an Athlon X4 830 (pure CPU) with **Q-learning self-play**. The AI played tens of thousands of games against itself and learned a non-losing strategy — against a perfect player it always draws (undefeated).

## What each file does

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher — just double-click this** |
| `tictactoe_train.py` | Lets the AI self-play to train, saves the experience as `ttt.npz` weights |
| `tictactoe_play.py` | Play against the trained AI |
| `bench.py` | Performance test: AI vs perfect player / vs random, tallies wins/losses |
| `ttt.npz` | The trained AI weights (this is what plays) |
| `tt_info.txt` | Training record (how many games, wins/losses/draws) |

## How to use

1. Double-click `menu.bat`.
2. Choose **2 Play vs AI** → you are X (first), input 1–9 to place, 0 to exit.
3. To retrain: choose **1**, input the number of games (default 50000).

## Notes

- `menu.bat` hard-codes `D:\TicTacToe Beta 0.1` — if you extract elsewhere, edit the `cd /d` line at the top.
- Requires Python 3.12 + numpy.
