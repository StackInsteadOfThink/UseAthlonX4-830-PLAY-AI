# -*- coding: utf-8 -*-
"""
毒舌/种族歧视检测 —— 训练脚本
- 中文: 字符级+二元组特征, 英文: 词级+二元组特征
- 分类器: softmax 逻辑回归 + TF-IDF
- 产物: model_zh.npz / model_en.npz  (3类: 正常/脏话/歧视)
"""
import re
import numpy as np

LABELS = ["正常", "脏话", "歧视"]

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
    vocab = []; seen = set()
    for t in texts:
        for tok in tokens_fn(t):
            if tok not in seen:
                seen.add(tok); vocab.append(tok)
    return vocab

def train_lang(texts, labels, tokens_fn, lr=0.5, iters=800, reg=0.01, seed=0):
    np.random.seed(seed)
    vocab = build_vocab(texts, tokens_fn)
    idx = {v: i for i, v in enumerate(vocab)}
    D = len(vocab); C = len(LABELS); N = len(texts)
    df = np.zeros(D)
    for t in texts:
        for tok in set(tokens_fn(t)):
            if tok in idx: df[idx[tok]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1.0
    X = np.zeros((N, D))
    for i, t in enumerate(texts):
        for tok in tokens_fn(t):
            if tok in idx: X[i, idx[tok]] += 1
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
        W -= lr * grad; b -= lr * gradb
        if (it + 1) % 200 == 0:
            acc = (probs.argmax(1) == labels_arr).mean()
            print(f"    iter {it+1}: 训练准确率 {acc:.3f}")
    acc = ((X @ W + b).argmax(1) == labels_arr).mean()
    print(f"    最终训练准确率: {acc:.3f}  特征维度: {D}")
    return W, b, vocab, idf

def main():
    zh = np.load('train_zh.npy', allow_pickle=True)
    en = np.load('train_en.npy', allow_pickle=True)
    zh_texts = [s[0] for s in zh]; zh_labels = [int(s[1]) for s in zh]
    en_texts = [s[0] for s in en]; en_labels = [int(s[1]) for s in en]
    print(f"中文语料 {len(zh_texts)} 条, 英文语料 {len(en_texts)} 条")
    print("训练中文模型...")
    Wz, bz, vz, idfz = train_lang(zh_texts, zh_labels, zh_tokens)
    np.savez('model_zh.npz', W=Wz, b=bz, vocab=np.array(vz, dtype=object), idf=idfz)
    print("训练英文模型...")
    We, be, ve, idfe = train_lang(en_texts, en_labels, en_tokens)
    np.savez('model_en.npz', W=We, b=be, vocab=np.array(ve, dtype=object), idf=idfe)
    print("完成! 已保存 model_zh.npz / model_en.npz")

if __name__ == '__main__':
    main()
