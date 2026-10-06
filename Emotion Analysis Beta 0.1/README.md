> 🌐 **如果你是外国用户 / English speakers**: 请阅读英文版说明 → [**README-en.md**](README-en.md) *(English version)*

# 情绪分析

中英文都能识别的 **6 种基本情绪**分类器：快乐 / 悲伤 / 愤怒 / 恐惧 / 惊讶 / 厌恶。输入一句话、一段话甚至整篇文章，它告诉你这段话表达什么情绪。在速龙 X4 830（纯 CPU）上训练。

## 文件都是干嘛的

| 文件 | 作用 |
|---|---|
| `menu.bat` | **启动器，双击这个就行** |
| `build_data.py` | 构建训练语料（中英文各几百句，标好情绪标签）|
| `train.py` | 训练中英文情绪模型，存成 `model_zh.npz` / `model_en.npz` |
| `predict.py` | 输入一句话/文章，识别情绪（自动判断中英文）|
| `bench.py` | 性能测试脚本 |
| `model_zh.npz` / `model_en.npz` | 中文 / 英文模型权重 |
| `train_zh.npy` / `train_en.npy` | 训练用的语料数据 |

## 怎么用

1. 双击 `menu.bat`。
2. 选 **2 情绪分析** → 输入一句话/一段话/一篇文章（中英文均可），输入 0 退出。
3. 想重新训练：选 **1**。

示例输入 `My flight got cancelled.` → 会识别为"悲伤"。

## 注意

- `menu.bat` 里写死了 `D:\Emotion Analysis Beta 0.1` 路径，解压到别处要先改开头的 `cd /d`。
- 需要 Python 3.12 + numpy。
