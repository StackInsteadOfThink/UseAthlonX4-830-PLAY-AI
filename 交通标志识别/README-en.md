# 🚧 Traffic Sign Recognition · 8 Countries, 93% Super Project

> **I'm training AI on a ¥25 Athlon X4 830 CPU — no GPU, just pure CPU.** Training data, the model itself, and all scripts will be released together. Please be patient.

## 🎯 What this is
An **8-country traffic sign recognition** system trained **purely on CPU, pure numpy, zero GPU**:

| Country | Status |
|---|---|
| 🇨🇳 China | Data collection / cleaning |
| 🇩🇪 Germany | Pilot done (v3 val 43.82%) |
| 🇯🇵 Japan | Data collection / cleaning |
| 🇰🇷 Korea | Data complete |
| 🇬🇧 UK | Data collection / cleaning |
| 🇺🇸 USA | Data collection / cleaning |
| 🇫🇷 France | Data collection / cleaning |
| 🇪🇸 Spain | Data collection / cleaning |

**Goal: push all 8 countries to 93% accuracy.**

## 🧱 Tech stack
- **CPU**: AMD Athlon X4 830 (2014 quad-core, ¥25 loose chip)
- **GPU**: None. Pure CPU compute.
- **Framework**: Hand-written CNN in pure `numpy` (Conv + BatchNorm + GAP + FC)
- **Data**: Scraped from 12 major platforms + human-reviewed one by one

## 🌐 Where the data comes from
Baidu + Bing + Douyin + Kuaishou + Bilibili + Xiaohongshu + Weibo + Zhihu + Baidu Tieba + Toutiao + Tencent + NetEase + Phoenix ... **12 super-platforms**, every source image manually reviewed, contamination (anime, news, racing, people) removed.

## 💪 Why I dare to brag
This **¥25 CPU** already trained a **3M-parameter LLM** (38 hours, pure CPU) in a previous project. Now it's going one bigger — **8-country traffic sign recognition**.

> The romance of a dumpster-diver: use the least material to do the wildest work. **Chinese people can fly.**

---

### My other AI projects
Look in the other folders of this repo (CIFAR-10, OCR, Q-Learning maze, Tic-Tac-Toe, emotion analysis, toxic detection, MiniChat 3M-param LLM, etc.), each with its own README.
