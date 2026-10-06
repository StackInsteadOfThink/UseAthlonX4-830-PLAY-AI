# -*- coding: utf-8 -*-
"""
情绪分析 —— 推理脚本
- 输入一段话/一句话/一篇文章 (中英文均可)
- 自动检测语言, 加载对应模型, 输出六种情绪的概率
"""
import re
import numpy as np

LABELS = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶"]

def zh_tokens(text):
    """中文特征: 中文字符 + 字符二元组"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    toks = list(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i+1])
    return toks

def en_tokens(text):
    """英文特征: 单词 + 单词二元组"""
    words = re.findall(r"[a-z']+", text.lower())
    toks = list(words)
    for i in range(len(words) - 1):
        toks.append(words[i] + ' ' + words[i+1])
    return toks

def featurize(text, tokens_fn, idx, idf):
    x = np.zeros(len(idx))
    for tok in tokens_fn(text):
        if tok in idx:
            x[idx[tok]] += 1
    x *= idf
    return x

def detect_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return 'zh' if zh > 0 else 'en'   # 有中文字符按中文, 否则英文

def load_model(path):
    d = np.load(path, allow_pickle=True)
    vocab = list(d['vocab'])
    idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

def predict(text, W, b, idx, idf, tokens_fn):
    x = featurize(text, tokens_fn, idx, idf)
    logits = x @ W + b
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs

def main():
    Wz, bz, idxz, idfz = load_model('model_zh.npz')
    We, be, idxe, idfe = load_model('model_en.npz')
    print("=" * 46)
    print("  情绪分析   (速龙 X4 830 训练)  六种基本情绪")
    print("=" * 46)
    print("输入一段话/一句话/文章(中英文均可), 输入 0 退出\n")
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
        probs = predict(text, W, b, idx, idf, tf)
        order = np.argsort(probs)[::-1]
        top = order[0]
        print(f"\n  情绪: 【{LABELS[top]}】  (概率 {probs[top]*100:.1f}%)")
        bars = []
        for i in order:
            pct = probs[i] * 100
            bar = '█' * int(pct // 5)
            bars.append(f"    {LABELS[i]:<2} {pct:5.1f}%  {bar}")
        print("\n".join(bars))
        print()

if __name__ == '__main__':
    main()
