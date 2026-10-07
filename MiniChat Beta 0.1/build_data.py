# -*- coding: utf-8 -*-
"""
MiniChat Beta 0.1 —— 语料预处理
读取 TinyShakespeare 文本, 构建字符级词表, 转成整数数组存 npz
"""
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tinyshakespeare.txt")

text = open(SRC, encoding="utf-8").read()

# 字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

data = np.array([stoi[c] for c in text], dtype=np.uint16)

# 9:1 切训练/验证
n = int(0.9 * len(data))
train, val = data[:n], data[n:]

np.savez(
    os.path.join(HERE, "data.npz"),
    train=train, val=val,
    vocab_size=vocab_size,
    chars="".join(chars),
)
print("词表大小:", vocab_size)
print("总字符数:", len(data))
print("训练字符:", len(train), "验证字符:", len(val))
print("字符集:", "".join(chars))
