> 🚀 **The author is currently working on a super traffic-sign-recognition project. Training data, the model, and all scripts will be released together — please be patient!**

# 🐎 Playing AI on an Athlon X4 830 — Pure-Numpy CIFAR-10 Image Classification

> *This is the English version of the main README. For the Chinese version, see [README.md](README.md).*

> An image-classification model trained on a **2015 AMD Athlon X4 830** (no AVX2, no GPU acceleration).
> Everything is **pure numpy** — hand-written convolution, pooling and backprop — to train a CNN from scratch that can recognize images.

---

## 📊 Results at a Glance

| Item | Result |
|---|---|
| Test accuracy | **71.63%** |
| Training time | ~105 min (20 epochs) |
| Model size | **~1 MB only** (269k parameters) |
| Device | Pure CPU, no GPU |

---

## 🎭 Fun Example (the epic fail)

The model knows only 10 classes: **airplane / car / bird / cat / deer / dog / frog / horse / ship / truck**.
It can't say "I don't know" — **anything outside those 10, it confidently guesses one anyway** — and that's where the comedy comes in.

**Recognition example** (author's own webcam shot of his desk):

![Recognition example: the garbage-tier desk](示例_垃圾佬桌面识别成船.jpg)

**Recognition result** (real output by dragging the image into `menu.bat`):

![Recognition result: ship 78.7%](示例_识别结果截图.jpg)

> **Author-tested**: I pointed my webcam at my desk (windowsill + fan + HDD + speaker).
> It confidently told me: **this is a ship, 78.7% confidence.**
> Why? Light-colored wall + a dark object in the middle, shrunk to 32×32, looks exactly like "a ship on the water".
> To make it recognize your room stuff, you'd need an ImageNet-class model + GPU — not an Athlon + pure numpy.

---

## 📦 How Much Data

| Item | Count |
|---|---|
| Classes | **10** (airplane/car/bird/cat/deer/dog/frog/horse/ship/truck) |
| **Training images per class** | **5,000** |
| **Test images per class** | **1,000** |
| **Total training set** | **50,000** |
| **Total test set** | **10,000** |
| **All images** | **60,000** |
| Single image size | 32 × 32 px, color (3-channel RGB) |
| Archive size | ~170 MB (public CIFAR-10 dataset) |

> Source: CIFAR-10 (public dataset, MIT license), used by researchers worldwide.

---

## 📁 What's in the Repo

| File | What it does |
|---|---|
| **`cifar_cnn_v2.py`** | **Training script**. Feeds 50k images to the network, trains the model (with data augmentation), saves it as `.npz`. |
| **`cifar_predict.py`** | **Recognition script**. Loads the model, recognizes any image you give it, prints "what it is + per-class probabilities". |
| **`download_data.py`** | **Download script**. Fetches CIFAR-10 from a mirror and extracts (~170 MB), run once. |
| **`cifar_model_v2.npz`** | **The trained model** (the finished product). None of the `.py` files are the "brain" — this is. |
| `README.md` | The doc you're reading. |

### 🧠 So what is a `.npz` file?

`.npz` is **numpy's compressed archive format** (Numpy Compressed Archive), designed to pack multiple numeric arrays into one file with automatic compression.

`cifar_model_v2.npz` holds the **8 arrays** produced by training:

```
W1/b1  →  1st conv layer weights + bias
W2/b2  →  2nd conv layer weights + bias
W3/b3  →  1st fully-connected weights + bias
W4/b4  →  2nd fully-connected weights + bias
```

That's **268,650 parameters**, stored as float32 (4 bytes each) → about 1 MB.
Plainly: **training is just adjusting these 269k numbers until they "recognize" 10 object types.** `.npz` is the save file of those numbers.

---

## ⏱ Time Spent

| Stage | Duration |
|---|---|
| Download CIFAR-10 (170MB) | ~2 min (mirror) |
| Train 20 epochs | **~105 min (1 h 45 min)** |
| Recognize one image | < 1 sec |

---

## 🖥 This Machine

| Part | Model |
|---|---|
| CPU | AMD Athlon X4 830 (2015, no AVX2) |
| RAM | ~8 GB usable |
| GPU | **Completely unused** during training (pure CPU) |
| OS | Windows 10 x64 |

Because there's no AVX2, PyTorch / TensorFlow won't even install — so **the whole network, conv, pooling and backprop are hand-written numpy**.

---

## 📢 About This Project (honest note)

This project was built with **Doubao AI (ByteDance) local-workflow assistance**.

If a `.py` script flashes and closes when you run it, **the script isn't broken** — usually:
1. `.py` files **can't be double-clicked** (it flashes and exits). Run them with `python` from the command prompt;
2. The environment isn't met (requires **Windows 10 64-bit + Python 3.12.6**).

**Recommended: use the packaged EXE version** — double-click and go, no Python install needed.

---

## 🚀 How to Use (idiot-proof)

> Open the command prompt: press **Win + R**, type `cmd`, Enter.
> Three parts: **①one-time setup**、**②daily inference**、**③optional training**.

### ① One-time setup (only once)

**Install dependencies (if NumPy and Pillow aren't installed, use pip)**
```
python -m pip install numpy pillow
```
> 💡 If you see `No module named 'numpy'` or `No module named 'PIL'`,
> your Python lacks these two libraries — the pip command above fixes it.

**Download data (~170 MB)**
```
python download_data.py
```

### ② Daily use · Recognize one image (★most common)

```
python cifar_predict.py image_path
```
Drag the image file **straight into the command window**, Enter. Or type the full path:
```
python cifar_predict.py D:\photos\cat.jpg
```
Example output:
```
Recognized as: cat  (confidence 62.2%)
--- Per-class probabilities ---
  airplane: 0.7%  car: 0.1%  bird: 0.3%
  cat: 62.2%  deer: 0.3%  dog: 36.0%  ...
```

### ③ Optional · Train your own model (skip if you only want inference)

```
python cifar_cnn_v2.py 20
```
The repo already ships a trained `cifar_model_v2.npz` — **if you just want inference, skip this step** and use the ready model. Run this only if you want to retrain from scratch.

---

## 🔜 What's Next (from the author)

This is **version one**, thanks for your patience.

The author will keep optimizing and ship a **one-click EXE** — then **you won't even need Python installed**, just download and **double-click to run**, recognize images, friendlier for beginners.

Version one is here first — **please be patient**, and feel free to open Issues / suggestions to push the author to update! 🚀

---

## 🏷 Classes

airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

---

## 💡 Lessons Learned

1. **No AVX2? Don't give up on training** — hand-written numpy still trains, just slower. This Athlon proves an old CPU can go from zero to image classification.
2. **Data augmentation is worth it** — adding horizontal flip lifted accuracy from 67% to 71.6%, better generalization, zero extra training cost.
3. **Low resolution is the hard ceiling** — at 32×32, contour-similar classes like bird/deer/frog naturally get confused. It's not under-training; it's just how much info the input has.
4. **Models can be shockingly small** — 269k parameters is enough for 10 classes. Big models are big because they learn a lot; for small tasks, a small net is the optimal answer.
5. **Hand-written backprop: watch the pooling layer** — pooling backprop splits the gradient evenly "one block → back to all"; misaligned shapes crash. Every pitfall was tuition.

---

## 🙏 Thanks

Thank you for reading this far. This 2015 old Athlon is still burning its remaining life — being able to train AI is already its greatest pride.
If you think it's fun, **Star / share / open an Issue** are all welcome. **Thanks for the support!**

---

## ⚠️ System Requirements (serious note)

A joke first: if you're still on a **floppy disk**, or a **Windows 95 / 98** machine — they're **all usable**... (strikethrough) actually **completely unusable**. 😄

For real: this project is built on **Python 3.12.6** and needs **Windows 10 (64-bit)** or newer. Make sure your system meets this before you start.

---

## 📂 Other AI Projects in This Repo

This repo isn't just CIFAR image classification — **it's a whole AI series trained on the same Athlon X4 830 with pure numpy**. There are more project folders in this repo; open them and read each one's `README.md` to learn more:

| Folder | Project |
|---|---|
| `Emotion Analysis Beta 0.1/` | CN/EN 6-emotion sentiment analysis |
| `OCR 数字 Beta 0.1/` | Handwritten digit recognition (MNIST) |
| `OffenseDetect Beta 0.1/` | Profanity + racial discrimination detection |
| `Q-Learning Beta 0.1/` | Q-learning maze solver |
| `TicTacToe Beta 0.1/` | TicTacToe AI |
| `ToxicDetection Beta 0.1/` | Toxic / discrimination detection |

(CIFAR-10 image classification lives in this repo's root — the page you're reading.)

Every project also has a downloadable zip in **Releases**. **Star / follow this repo to see how much AI a 2015 Athlon can really play with.**

---

*This repo uses the public CIFAR-10 dataset (MIT license).*
