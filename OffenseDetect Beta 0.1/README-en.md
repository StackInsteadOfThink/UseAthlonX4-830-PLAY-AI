# Profanity + Racial Discrimination Detection (granular version)

A finer-grained detector than "toxic": **one model judges "is it profanity", another judges "which group is discriminated"** (none / white / yellow / black). One set each for CN and EN — 4 models total — trained on an Athlon X4 830 (pure CPU).

## What each file does

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher — just double-click this** |
| `build_data.py` | Builds the training corpus (profanity + discrimination data, one set each in CN/EN) |
| `train.py` | Trains the 4 models |
| `predict.py` | Input text, outputs both "is profanity" and "discriminated group" judgments |
| `bench.py` | Performance test script |
| `abuse_zh.npz` / `abuse_en.npz` | Chinese / English "profanity or not" models |
| `discr_zh.npz` / `discr_en.npz` | Chinese / English "discriminated group" models |
| `abuse_zh.npy` / `abuse_en.npy` etc. | Corresponding corpus data (.npy) |

## How to use

1. Double-click `menu.bat`.
2. Choose **2 Detect** → type text (CN or EN), it auto-judges "profanity + discriminated group", type 0 to exit.
3. To retrain: choose **1** (trains all 4 models).

Example: `you nigger.` → judged "profanity" and "discriminates black people".

## Notes

- The two judgments are separate models, independent of each other.
- `menu.bat` hard-codes `D:\OffenseDetect Beta 0.1` — if you extract elsewhere, edit the `cd /d` line at the top.
- Requires Python 3.12 + numpy.
