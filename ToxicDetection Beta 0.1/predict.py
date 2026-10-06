# -*- coding: utf-8 -*-
"""
毒舌/种族歧视检测 —— 推理脚本
输入一句话, 判定: 正常 / 脏话 / 种族歧视 (中英文)
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

def featurize(text, tf, idx, idf):
    x = np.zeros(len(idx))
    for tok in tf(text):
        if tok in idx: x[idx[tok]] += 1
    return x * idf

def detect_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return 'zh' if zh > 0 else 'en'

def load_model(p):
    d = np.load(p, allow_pickle=True)
    vocab = list(d['vocab']); idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

def main():
    Wz, bz, idxz, idfz = load_model('model_zh.npz')
    We, be, idxe, idfe = load_model('model_en.npz')
    print("=" * 50)
    print("  毒舌/种族歧视检测  (速龙 X4 830 训练)")
    print("  判定: 正常 / 脏话 / 种族歧视  (中英文)")
    print("=" * 50)
    print("输入一句话, 输入 0 退出\n")
    while True:
        try:
            text = input("请输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出"); break
        if text in ('0', '退出', 'q', 'quit'):
            print("已退出"); break
        if not text:
            continue
        if detect_lang(text) == 'zh':
            W, b, idx, idf, tf = Wz, bz, idxz, idfz, zh_tokens
        else:
            W, b, idx, idf, tf = We, be, idxe, idfe, en_tokens
        p = np.exp(featurize(text, tf, idx, idf) @ W + b)
        p = np.exp(p - p.max()); p /= p.sum()
        o = np.argsort(p)[::-1]
        top = o[0]
        mark = ""
        if LABELS[top] == "脏话":
            mark = "  ⚠️ 检测到脏话/冒犯用语"
        elif LABELS[top] == "歧视":
            mark = "  🚫 检测到种族歧视言论"
        print(f"\n  判定: 【{LABELS[top]}】 (概率 {p[top]*100:.0f}%){mark}")
        bars = []
        for i in o:
            pct = p[i] * 100
            bars.append(f"    {LABELS[i]:<2} {pct:5.1f}%  {'█'*int(pct//5)}")
        print("\n".join(bars)); print()

if __name__ == '__main__':
    main()
