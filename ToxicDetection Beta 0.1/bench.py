# -*- coding: utf-8 -*-
"""毒舌/歧视检测验证: 训练集准确率 + 测句"""
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
def featurize(text, tf, idx, idf):
    x = np.zeros(len(idx))
    for tok in tf(text):
        if tok in idx: x[idx[tok]] += 1
    return x * idf
def load_model(p):
    d = np.load(p, allow_pickle=True)
    vocab = list(d['vocab']); idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

zh = np.load('train_zh.npy', allow_pickle=True)
en = np.load('train_en.npy', allow_pickle=True)
Wz, bz, idxz, idfz = load_model('model_zh.npz')
We, be, idxe, idfe = load_model('model_en.npz')
zc = 0
for s in zh:
    p = np.exp(featurize(s[0], zh_tokens, idxz, idfz) @ Wz + bz)
    p = np.exp(p - p.max()); p /= p.sum()
    if p.argmax() == int(s[1]): zc += 1
ec = 0
for s in en:
    p = np.exp(featurize(s[0], en_tokens, idxe, idfe) @ We + be)
    p = np.exp(p - p.max()); p /= p.sum()
    if p.argmax() == int(s[1]): ec += 1
print(f"中文训练集准确率: {zc}/{len(zh)} = {zc/len(zh)*100:.1f}%")
print(f"英文训练集准确率: {ec}/{len(en)} = {ec/len(en)*100:.1f}%\n")

def show(text, W, b, idx, idf, tf):
    p = np.exp(featurize(text, tf, idx, idf) @ W + b)
    p = np.exp(p - p.max()); p /= p.sum()
    o = np.argsort(p)[::-1]
    mark = "  正常" if LABELS[o[0]]=="正常" else ("  [脏话]" if LABELS[o[0]]=="脏话" else "  [歧视]")
    print(f"  「{text}」 -> {LABELS[o[0]]} {p[o[0]]*100:.0f}%{mark}  | " + ", ".join(f"{LABELS[i]}:{p[i]*100:.0f}%" for i in o))
    return LABELS[o[0]]

print("中文测句:")
show("今天天气不错，出去走走。", Wz, bz, idxz, idfz, zh_tokens)
show("你是个白痴。", Wz, bz, idxz, idfz, zh_tokens)
show("你是一个香蕉", Wz, bz, idxz, idfz, zh_tokens)
show("你这个黑鬼。", Wz, bz, idxz, idfz, zh_tokens)
show("黄皮猴子。", Wz, bz, idxz, idfz, zh_tokens)
show("周末去爬山。", Wz, bz, idxz, idfz, zh_tokens)
print()
print("英文测句:")
show("The weather is nice today.", We, be, idxe, idfe, en_tokens)
show("you idiot.", We, be, idxe, idfe, en_tokens)
show("you banana.", We, be, idxe, idfe, en_tokens)
show("go back to your country.", We, be, idxe, idfe, en_tokens)
show("I like this book.", We, be, idxe, idfe, en_tokens)
print()
