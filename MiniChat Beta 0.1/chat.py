# -*- coding: utf-8 -*-
"""
MiniChat Beta 0.1 —— 聊天推理
加载训练好的 minichat.npz, 英文对话。温度1, 上下文窗口=训练时的block_size(128字符)。
"""
import numpy as np, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "minichat.npz")

if not os.path.exists(CKPT):
    print("没找到训练权重 minichat.npz，请先运行 train.py 训练。")
    sys.exit(1)

d = np.load(CKPT)
P = {k: d[k] for k in d.files if k != "iters"}
vocab_size = P["wte"].shape[0]
n_embd = P["wte"].shape[1]
train.CFG["n_embd"] = n_embd
train.CFG["n_head"] = 4
train.CFG["n_layer"] = sum(1 for k in P if k.startswith("attn_0_"))
train.CFG["block_size"] = P["wpe"].shape[0]
chars = train.chars
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
BLOCK = train.CFG["block_size"]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

def generate(prompt, max_new=90, temperature=1.0):
    toks = encode(prompt)
    if len(toks) > BLOCK:
        toks = toks[-BLOCK:]
    idx = list(toks)
    for _ in range(max_new):
        x = idx[-BLOCK:]
        logits, _ = train.forward(P, np.array([x], dtype=np.int64))
        logit = logits[0, -1]
        probs = softmax(logit / temperature) if temperature != 1.0 else softmax(logit)
        nxt = int(np.random.choice(vocab_size, p=probs))
        idx.append(nxt)
    return decode(idx[len(toks):])

def main():
    print("=" * 52)
    print("  MiniChat Beta 0.1  (300万参数 · 英文对话 · 速龙 X4 830)")
    print("  输入英文聊天, 回车发送 | 输入 exit / quit / 0 退出")
    print("  注意: 纯CPU推理较慢, 请耐心等待; 上下文窗口 128 字符")
    print("=" * 52)
    while True:
        try:
            user = input("\n你> ")
        except EOFError:
            break
        if not user.strip():
            continue
        if user.strip().lower() in ("exit", "quit", "0", "q"):
            print("拜拜~")
            break
        t0 = time.time()
        print("AI> ", end="", flush=True)
        try:
            reply = generate(user.strip())
            print(reply)
        except Exception as e:
            print(f"(生成出错: {e})")
        print(f"[耗时 {time.time()-t0:.1f} 秒]")

if __name__ == "__main__":
    main()
