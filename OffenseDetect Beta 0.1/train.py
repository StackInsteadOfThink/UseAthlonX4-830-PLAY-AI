# -*- coding: utf-8 -*-
"""
脏话检测 + 种族歧视检测 —— 训练脚本
- 中文: 字符级+bigram, 英文: 词级+bigram
- TF-IDF + softmax 逻辑回归
- 4 个模型: abuse_zh/abuse_en(二分类) discr_zh/discr_en(四分类)
- 英文迭代次数更多(训练强度高)
产物: abuse_zh.npz abuse_en.npz discr_zh.npz discr_en.npz
"""
import re
import numpy as np

ABUSE_LABELS = ["正常", "脏话"]
DISC_LABELS = ["无歧视", "歧视白人", "歧视黄种人", "歧视黑种人"]

def zh_tokens(text):
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    toks = list(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i+1])
    return toks

def en_tokens(text):
    words = re.findall(r"[a-z']+", text.lower())
    toks = list(words)
    for i in range(len(words) - 1):
        toks.append(words[i] + ' ' + words[i+1])
    return toks

def build_vocab(texts, tokens_fn):
    vocab, seen = [], set()
    for t in texts:
        for tok in tokens_fn(t):
            if tok not in seen:
                seen.add(tok); vocab.append(tok)
    return vocab

def train_lang(texts, labels, tokens_fn, iters, lr=0.5, reg=0.01, seed=0):
    np.random.seed(seed)
    vocab = build_vocab(texts, tokens_fn)
    idx = {v: i for i, v in enumerate(vocab)}
    D = len(vocab); N = len(texts)
    C = len(set(labels))
    df = np.zeros(D)
    for t in texts:
        for tok in set(tokens_fn(t)):
            if tok in idx:
                df[idx[tok]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1.0
    X = np.zeros((N, D))
    for i, t in enumerate(texts):
        for tok in tokens_fn(t):
            if tok in idx:
                X[i, idx[tok]] += 1
        X[i] *= idf
    W = np.random.randn(D, C) * 0.01
    b = np.zeros(C)
    Y = np.eye(C)[np.array(labels)]
    labels_arr = np.array(labels)
    for it in range(iters):
        logits = X @ W + b
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad = (X.T @ (probs - Y)) / N + reg * W
        gradb = (probs - Y).mean(axis=0)
        W -= lr * grad
        b -= lr * gradb
        if (it + 1) % 200 == 0:
            acc = (probs.argmax(1) == labels_arr).mean()
            print(f"    iter {it+1}: 训练准确率 {acc:.3f}")
    acc = ((X @ W + b).argmax(1) == labels_arr).mean()
    print(f"    最终训练准确率: {acc:.3f}  特征维度: {D}  类别数: {C}")
    return W, b, vocab, idf

def run(name, fname, texts, labels, tokens_fn, labels_list, iters):
    print(f"训练 {name} ...")
    W, b, vocab, idf = train_lang(texts, labels, tokens_fn, iters=iters)
    np.savez(fname, W=W, b=b, vocab=np.array(vocab, dtype=object), idf=idf,
             labels=np.array(labels_list, dtype=object))

def main():
    # 脏话(二分类)
    az = np.load('abuse_zh.npy', allow_pickle=True)
    ae = np.load('abuse_en.npy', allow_pickle=True)
    tz = [s[0] for s in az]; lz = [int(s[1]) for s in az]
    te = [s[0] for s in ae]; le = [int(s[1]) for s in ae]
    run("脏话-中文", 'abuse_zh.npz', tz, lz, zh_tokens, ABUSE_LABELS, iters=600)
    run("脏话-英文", 'abuse_en.npz', te, le, en_tokens, ABUSE_LABELS, iters=1200)  # 英文强度更高
    # 歧视(四分类)
    dz = np.load('discr_zh.npy', allow_pickle=True)
    de = np.load('discr_en.npy', allow_pickle=True)
    tz2 = [s[0] for s in dz]; lz2 = [int(s[1]) for s in dz]
    te2 = [s[0] for s in de]; le2 = [int(s[1]) for s in de]
    run("歧视-中文", 'discr_zh.npz', tz2, lz2, zh_tokens, DISC_LABELS, iters=600)
    run("歧视-英文", 'discr_en.npz', te2, le2, en_tokens, DISC_LABELS, iters=1200)  # 英文强度更高
    print("全部训练完成! 已保存 4 个模型")

if __name__ == '__main__':
    main()
