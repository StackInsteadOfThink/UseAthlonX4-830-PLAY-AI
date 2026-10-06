# -*- coding: utf-8 -*-
"""情绪分析验证: 训练集准确率 + 手动测句"""
import re
import numpy as np

LABELS = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶"]

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
def featurize(text, tf, idx, idf):
    x = np.zeros(len(idx))
    for tok in tf(text):
        if tok in idx: x[idx[tok]] += 1
    x *= idf
    return x
def load_model(p):
    d = np.load(p, allow_pickle=True)
    vocab = list(d['vocab']); idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

# 1) 训练集准确率
zh = np.load('train_zh.npy', allow_pickle=True)
en = np.load('train_en.npy', allow_pickle=True)
Wz, bz, idxz, idfz = load_model('model_zh.npz')
We, be, idxe, idfe = load_model('model_en.npz')
zcorr = 0
for s in zh:
    text, lab = s[0], int(s[1])
    probs = np.exp((featurize(text, zh_tokens, idxz, idfz) @ Wz + bz) - 0)
    probs = np.exp(probs - probs.max()); probs /= probs.sum()
    if probs.argmax() == lab: zcorr += 1
ecorr = 0
for s in en:
    text, lab = s[0], int(s[1])
    probs = np.exp((featurize(text, en_tokens, idxe, idfe) @ We + be) - 0)
    probs = np.exp(probs - probs.max()); probs /= probs.sum()
    if probs.argmax() == lab: ecorr += 1
print(f"中文训练集准确率: {zcorr}/{len(zh)} = {zcorr/len(zh)*100:.1f}%")
print(f"英文训练集准确率: {ecorr}/{len(en)} = {ecorr/len(en)*100:.1f}%")
print()

# 2) 手动测句(模拟新输入)
def show(text, W, b, idx, idf, tf):
    probs = np.exp(featurize(text, tf, idx, idf) @ W + b)
    probs = np.exp(probs - probs.max()); probs /= probs.sum()
    o = np.argsort(probs)[::-1]
    print(f"  「{text}」")
    print(f"    -> {LABELS[o[0]]} ({probs[o[0]]*100:.0f}%)  完整: " + ", ".join(f"{LABELS[i]}{probs[i]*100:.0f}%" for i in o))
    return LABELS[o[0]]

print("中文测句:")
c1 = show("这部电影太无聊了，浪费时间，我讨厌它。", Wz, bz, idxz, idfz, zh_tokens)
c2 = show("终于见到你了，我好开心，感觉整个世界都亮了。", Wz, bz, idxz, idfz, zh_tokens)
c3 = show("我不敢一个人走夜路，黑漆漆的好害怕。", Wz, bz, idxz, idfz, zh_tokens)
c4 = show("他居然一声不吭就走了，我完全没想到。", Wz, bz, idxz, idfz, zh_tokens)
c5 = show("又被老板骂了一顿，这种委屈我受够了。", Wz, bz, idxz, idfz, zh_tokens)
c6 = show("想到远方的亲人，心里酸酸的，眼泪止不住。", Wz, bz, idxz, idfz, zh_tokens)
print()
print("英文测句:")
e1 = show("I'm so excited, this is the happiest day of my life!", We, be, idxe, idfe, en_tokens)
e2 = show("He lied to me again, I'm absolutely furious.", We, be, idxe, idfe, en_tokens)
e3 = show("I'm terrified of the dark, my heart is pounding.", We, be, idxe, idfe, en_tokens)
e4 = show("Wow, I never expected that, I'm completely shocked.", We, be, idxe, idfe, en_tokens)
e5 = show("The food tastes awful, I want to throw up.", We, be, idxe, idfe, en_tokens)
e6 = show("I feel so alone and depressed, nothing matters anymore.", We, be, idxe, idfe, en_tokens)
print()
