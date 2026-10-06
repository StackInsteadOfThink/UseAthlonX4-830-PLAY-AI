# Sentiment Analysis

A **6-basic-emotion** classifier that works in both Chinese and English: happy / sad / angry / fearful / surprised / disgusted. Feed it a sentence, a paragraph or even a whole article, and it tells you the emotion. Trained on an Athlon X4 830 (pure CPU).

## What each file does

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher — just double-click this** |
| `build_data.py` | Builds the training corpus (a few hundred sentences each in CN/EN, labeled with emotions) |
| `train.py` | Trains the CN + EN sentiment models, saves `model_zh.npz` / `model_en.npz` |
| `predict.py` | Input a sentence/article, detect the emotion (auto-detects CN or EN) |
| `bench.py` | Performance test script |
| `model_zh.npz` / `model_en.npz` | Chinese / English model weights |
| `train_zh.npy` / `train_en.npy` | Training corpus data |

## How to use

1. Double-click `menu.bat`.
2. Choose **2 Sentiment Analysis** → type a sentence / paragraph / article (CN or EN), type 0 to exit.
3. To retrain: choose **1**.

Example: `My flight got cancelled.` → classified as "sad".

## Notes

- `menu.bat` hard-codes `D:\Emotion Analysis Beta 0.1` — if you extract elsewhere, edit the `cd /d` line at the top.
- Requires Python 3.12 + numpy.
