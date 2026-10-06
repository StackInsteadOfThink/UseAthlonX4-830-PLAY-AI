> 🌐 **如果你是外国用户 / English speakers**: 请阅读英文版说明 → [**README-en.md**](README-en.md) *(English version)*

# 三子棋 AI (TicTacToe)

在速龙 X4 830（纯 CPU）上用 **Q-learning 自对弈**训练的井字棋 AI。AI 自己和自己下了几万局，学会了不输的下法——它对完美棋手能全平局（不败）。

## 文件都是干嘛的

| 文件 | 作用 |
|---|---|
| `menu.bat` | **启动器，双击这个就行** |
| `tictactoe_train.py` | 让 AI 自对弈训练，把经验存成 `ttt.npz` 权重 |
| `tictactoe_play.py` | 和训练好的 AI 下棋 |
| `bench.py` | 性能测试：AI vs 完美棋手 / vs 随机，统计胜负 |
| `ttt.npz` | 训练好的 AI 权重（下棋就用它）|
| `tt_info.txt` | 训练数据记录（训练了多少局、胜负平）|

## 怎么用

1. 双击 `menu.bat`。
2. 选 **2 和 AI 下棋** → 你执 X（先手），输入 1–9 落子，0 退出。
3. 想重新训练：选 **1**，输入局数（默认 50000）。

## 注意

- `menu.bat` 里写死了 `D:\TicTacToe Beta 0.1` 路径，解压到别处要先改开头的 `cd /d`。
- 需要 Python 3.12 + numpy。
