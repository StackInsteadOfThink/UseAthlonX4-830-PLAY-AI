# MiniChat Beta 0.1 —— 300 万参数英文小模型（纯 numpy / 速龙 CPU）

在 **速龙 X4 830（四核纯 CPU，零依赖 numpy）** 上从零训练的一个 **300 万参数字符级 transformer**。项目名为 MiniChat。

> 🌍 If you are an English-speaking user, please read [README-en.md](README-en.md) for the English version.

## ⚠️ 诚实说明（先看这个）

**这个模型不会真正对话。** 实测用「你好」「你叫什么名字」「如果你有一张纸，你会干什么」提问，它输出的都是无意义字符（乱码）。原因：
- **300 万参数太小**——真正能对话的模型动辄几十亿参数，300 万连入门都不算
- **字符级 + 莎士比亚语料**——它学的是"预测下一个字符"，语料是文学剧本（不是对话），它根本不知道"对话"是什么
- 训练 loss 约 2.6（字符级，偏高），生成近乎随机

**它的价值**：证明了 **在速龙 X4 830 上能完整跑通 300 万参数 transformer 训练流程**——从零手写 numpy 实现、38 小时训练、权重可保存可加载。这是一个"全站第一例"式的硬核玩具，**不是**一个能聊天的 AI。

## 这是干什么的

用 TinyShakespeare（约 1MB 莎士比亚剧本）作为英文语料，训练一个小型字符级 transformer。你输入英文开头，它接下去生成"像英文"的字符流。请把它当垃圾佬的极限玩具看待。

## 文件说明

| 文件 | 作用 |
|---|---|
| `menu.bat` | **启动器**，双击进入菜单：`1=聊天推理` `2=重新训练` `3=退出` |
| `chat.py` | 聊天推理（加载 minichat.npz）|
| `train.py` | 训练脚本（纯 numpy 手写 transformer，含梯度检查 `--check`、断点续训）|
| `build_data.py` | 把 tinyshakespeare.txt 转成词表 + 整数数组 data.npz |
| `smoke.py` | 冒烟测试（小跑验证训练）|
| `test_grad.py` | 梯度诊断脚本 |
| `tinyshakespeare.txt` | 英文训练语料 |
| `data.npz` | 预处理后的训练数据 |
| `minichat.npz` | **训练好的模型权重（核心产物，12.2MB）**|

## 怎么用

**① 只想聊天推理**：双击 `menu.bat` → 选 `1` → 输入英文，回车发送，`exit` / `quit` / `0` 退出。
**② 想从零重新训练**：双击 `menu.bat` → 选 `2`（需 `data.npz` 存在，没有就先运行 `python build_data.py`）。训练需数天，纯 CPU 慢慢磨。

## 训练实况（实测）

- 迭代 **15000 步**，训练耗时约 **38 小时**（速龙 X4 830 纯 CPU）
- 最终 val_loss ≈ **2.64**
- 全程速度约 0.2 iter/s，一路爬升
- `train_log.txt` 实时写 loss / 速度 / 预计剩余

## 训练参数（凑到 300 万参数）

- 字符级词表 65，embedding 维度 256，4 层 transformer，4 头注意力，上下文窗口 128
- 参数总量约 **320 万**
- 优化：AdamW，交叉熵，梯度裁剪，batch=4，lr=3e-3
- `train.py --check` 通过完整数值梯度校验（最大相对误差 <1e-3）

## 注意

- 模型是**英文**（语料是英文剧本），请用英文输入；中文输入它完全不认识。
- 纯 CPU 推理慢：生成每条回复可能要几十秒到几分钟，请耐心。
- 需要 Python 3.12 + numpy。
- `menu.bat` 已用相对路径（`%~dp0`），解压到任何地方都能用，不用改路径。

## 版权与联系

- 模型权重、脚本、README 可自由使用（训练数据仅用于本项目，未分发原图）。
- 作者 GitHub：[StackInsteadOfThink](https://github.com/StackInsteadOfThink)
