# -*- coding: utf-8 -*-
"""
脏话 + 种族歧视检测 —— 推理脚本
输入一段文本, 自动检测语言, 同时输出:
  1. 脏话检测(是/否)
  2. 种族歧视检测(无/白/黄/黑)
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

def detect_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return 'zh' if zh > 0 else 'en'

def load_model(path):
    d = np.load(path, allow_pickle=True)
    vocab = list(d['vocab']); idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

def probs_of(text, W, b, idx, idf, tokens_fn):
    x = np.zeros(len(idx))
    for tok in tokens_fn(text):
        if tok in idx:
            x[idx[tok]] += 1
    x *= idf
    p = np.exp(x @ W + b); p = np.exp(p - p.max()); p /= p.sum()
    return p

def fmt_probs(labels, p):
    parts = []
    for i, lab in enumerate(labels):
        bar = '█' * int(p[i] * 100 // 5)
        parts.append(f"  {lab} {p[i]*100:5.1f}% {bar}")
    return "\n".join(parts)

def main():
    Waz, baz, idxaz, idfaz = load_model('abuse_zh.npz')
    Wae, bae, idxae, idfae = load_model('abuse_en.npz')
    Wdz, bdz, idxdz, idfdz = load_model('discr_zh.npz')
    Wde, bde, idxde, idfde = load_model('discr_en.npz')
    print("=" * 50)
    print("  脏话 + 种族歧视检测  (速龙 X4 830 训练)")
    print("=" * 50)
    print("输入文本(中英文均可), 输入 0 退出\n")
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
            Wab, bab, iab, idab, Wdi, bdi, idi, idid, tf = Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens
        else:
            Wab, bab, iab, idab, Wdi, bdi, idi, idid, tf = Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens
        pa = probs_of(text, Wab, bab, iab, idab, tf)
        pd = probs_of(text, Wdi, bdi, idi, idid, tf)
        a = pa.argmax(); d = pd.argmax()
        print(f"\n  文本: {text}")
        print(f"  ── 脏话判定: 【{ABUSE_LABELS[a]}】 (概率 {pa[a]*100:.0f}%)")
        print(fmt_probs(ABUSE_LABELS, pa))
        print(f"  ── 歧视判定: 【{DISC_LABELS[d]}】 (概率 {pd[d]*100:.0f}%)")
        print(fmt_probs(DISC_LABELS, pd))
        print()

if __name__ == '__main__':
    main()
