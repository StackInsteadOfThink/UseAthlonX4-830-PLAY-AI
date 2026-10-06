# Toxic / Racial Discrimination Detection

Judges a sentence into one of three classes: **normal / profanity / racial discrimination**. Supports Chinese and English, trained on an Athlon X4 830 (pure CPU).

## What each file does

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher — just double-click this** |
| `build_data.py` | Builds the training corpus (normal/profanity/discrimination, one set each in CN/EN) |
| `train.py` | Trains the CN + EN detection models |
| `predict.py` | Input a sentence, gives the three class probabilities |
| `bench.py` | Performance test script |
| `model_zh.npz` / `model_en.npz` | Chinese / English model weights |
| `train_zh.npy` / `train_en.npy` | Training corpus data |

## How to use

1. Double-click `menu.bat`.
2. Choose **2 Detect a sentence** → input CN or EN, type 0 to exit.
3. To retrain: choose **1**.

Example: `你是一个香蕉` ("you're a banana") → judged "discrimination".

## Notes

- A sentence may cross multiple lines at once; the output is three class probabilities and takes the highest.
- Some ambiguous critical sentences (e.g. "你这人素质真差" / "you have bad manners") may be misjudged — an inherent limitation of 3-class classification.
- `menu.bat` hard-codes `D:\ToxicDetection Beta 0.1` — if you extract elsewhere, edit the `cd /d` line at the top.
- Requires Python 3.12 + numpy.
