# -*- coding: utf-8 -*-
"""脏话与歧视检测验证: 测句"""
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

def load_model(p):
    d = np.load(p, allow_pickle=True)
    vocab = list(d['vocab']); idx = {v: i for i, v in enumerate(vocab)}
    return d['W'], d['b'], idx, d['idf']

def probs(text, W, b, idx, idf, tf):
    x = np.zeros(len(idx))
    for tok in tf(text):
        if tok in idx: x[idx[tok]] += 1
    x *= idf
    p = np.exp(x @ W + b); p = np.exp(p - p.max()); p /= p.sum()
    return p

def show(text, Wab, bab, iab, idab, Wdi, bdi, idi, idid, tf):
    pa = probs(text, Wab, bab, iab, idab, tf); a = pa.argmax()
    pd = probs(text, Wdi, bdi, idi, idid, tf); d = pd.argmax()
    print(f"  「{text}」")
    print(f"    脏话: {ABUSE_LABELS[a]} ({pa[a]*100:.0f}%) | 歧视: {DISC_LABELS[d]} ({pd[d]*100:.0f}%)")

Waz, baz, idxaz, idfaz = load_model('abuse_zh.npz')
Wae, bae, idxae, idfae = load_model('abuse_en.npz')
Wdz, bdz, idxdz, idfdz = load_model('discr_zh.npz')
Wde, bde, idxde, idfde = load_model('discr_en.npz')

print("中文测句:")
show("你个傻逼，滚远点。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
show("黑鬼滚回非洲去。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
show("洋鬼子滚出中国。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
show("小日本真讨厌。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
show("今天天气不错，想去散步。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
show("他成绩很好，我很佩服。", Waz, baz, idxaz, idfaz, Wdz, bdz, idxdz, idfdz, zh_tokens)
print()
print("英文测句:")
show("You're a piece of shit, shut up.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
show("Go back to Africa, you monkey.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
show("White people are arrogant trash.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
show("Chinks go back to China.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
show("The weather is nice today, let's walk.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
show("I really enjoyed the meal, thank you.", Wae, bae, idxae, idfae, Wde, bde, idxde, idfde, en_tokens)
