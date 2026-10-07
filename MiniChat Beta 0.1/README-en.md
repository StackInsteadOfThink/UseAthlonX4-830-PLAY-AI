# MiniChat Beta 0.1 — A 3M-Parameter English Model (Pure NumPy / Athlon CPU)

A **3-million-parameter character-level transformer** trained from scratch on an **AMD Athlon X4 830 (quad-core, pure CPU, zero-dependency NumPy)**. Project name: MiniChat.

> 🌍 中文版请阅读 [README.md](README.md)。

## ⚠️ Honest note (read this first)

**This model cannot actually chat.** Tested with prompts like "Hello", "What's your name?" and "If you had a piece of paper, what would you do?", it outputs only gibberish (meaningless characters). Why:
- **3M parameters is far too small** — real chatbots have billions of parameters.
- **Character-level + Shakespeare corpus** — it learns to "predict the next character" from literary drama, not dialogue. It has no idea what a conversation is.
- Training loss ≈ 2.6 (character-level, on the high side), so generation is near-random.

**Its real value**: it proves you can **complete a full 3M-parameter transformer training pipeline on an Athlon X4 830** — a from-scratch NumPy implementation, ~38 hours of training, saveable/loadable weights. It is a "first-of-its-kind on the whole site" hardcore toy, **not** a chatbot.

## What it does

Trained on TinyShakespeare (~1MB of Shakespeare's plays) as English corpus, this small character-level transformer takes an English prefix and continues with "English-like" character streams. Treat it as an extreme toy for the "junk-grabber" community.

## Files

| File | Purpose |
|---|---|
| `menu.bat` | **Launcher** — double-click opens a menu: `1=chat infer` `2=retrain` `3=exit` |
| `chat.py` | Chat inference (loads minichat.npz) |
| `train.py` | Training script (pure-NumPy transformer, gradient check `--check`, resume support) |
| `build_data.py` | Converts tinyshakespeare.txt into vocab + integer array data.npz |
| `smoke.py` | Smoke test (short run to verify training) |
| `test_grad.py` | Gradient diagnostic script |
| `tinyshakespeare.txt` | English training corpus |
| `data.npz` | Preprocessed training data |
| `minichat.npz` | **Trained model weights (core artifact, 12.2MB)** |

## How to use

**① Just chat/infer**: double-click `menu.bat` → choose `1` → type English, Enter to send, `exit`/`quit`/`0` to leave.
**② Retrain from scratch**: double-click `menu.bat` → choose `2` (needs `data.npz`; run `python build_data.py` first if missing). Training takes days on pure CPU.

## Training facts (measured)

- **15000 iterations**, ~**38 hours** on the Athlon X4 830 (pure CPU)
- Final val_loss ≈ **2.64**
- ~0.2 iter/s throughout, slowly climbing
- `train_log.txt` logs loss / speed / ETA in real time

## Architecture (3M params)

- Character vocab 65, embedding dim 256, 4-layer transformer, 4 heads, context window 128
- ~3.2M total parameters
- AdamW, cross-entropy, gradient clipping, batch=4, lr=3e-3
- `train.py --check` passes full numerical gradient check (max rel error <1e-3)

## Notes

- The model is **English-only** (English corpus). Chinese input is unrecognized.
- Pure-CPU inference is slow: each reply can take tens of seconds to minutes.
- Requires Python 3.12 + NumPy.
- `menu.bat` uses a relative path (`%~dp0`), so it works wherever you extract it.

## License & contact

- Weights, scripts and README are free to use (training data is project-internal; no original images redistributed).
- Author GitHub: [StackInsteadOfThink](https://github.com/StackInsteadOfThink)
