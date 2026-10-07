# -*- coding: utf-8 -*-
"""诊断: 对比解析梯度 vs 数值梯度, 定位错误模式"""
import numpy as np, math, sys
sys.path.insert(0, r"D:\MiniChat Beta 0.1")
import train

# 超小模型
train.CFG = dict(train.CFG, n_embd=8, n_head=2, n_layer=2, block_size=6, batch_size=2)
np.random.seed(1)
V = 12
train.vocab_size = V
P = train.init_params()
P = {k: v.astype(np.float64) for k, v in P.items()}
x = np.random.randint(0, V, (2, 6)).astype(np.int64)
y = np.random.randint(0, V, (2, 6)).astype(np.int64)
lg, C = train.forward(P, x)
loss, dl = train.cross_entropy(lg, y)
G = train.backward(P, C, dl)

eps = 1e-4
for k in ["attn_1_wq", "mlp_1_w1", "ln1_1_g", "lnf_g"]:
    W = P[k]
    num = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        Wp = P[k].copy(); Wp[idx] += eps; Pp = dict(P); Pp[k] = Wp
        l1, _ = train.cross_entropy(train.forward(Pp, x)[0], y)
        Wm = P[k].copy(); Wm[idx] -= eps; Pm = dict(P); Pm[k] = Wm
        l2, _ = train.cross_entropy(train.forward(Pm, x)[0], y)
        num[idx] = (l1 - l2) / (2*eps)
    ga = G[k]; gn = num
    # 采样几个元素看值
    print(f"== {k} (shape {W.shape}) ==")
    flat_a = ga.ravel(); flat_n = gn.ravel()
    for i in [0, 1, W.size//2, W.size-1]:
        print(f"  idx{i}: 解析={flat_a[i]:.4e}  数值={flat_n[i]:.4e}  比值解析/数值={flat_a[i]/flat_n[i] if flat_n[i]!=0 else 'inf'}")
    rel = np.abs(gn-ga)/(np.maximum(np.abs(gn),np.abs(ga))+1e-9)
    print(f"  最大相对误差={rel.max():.2e}, 平均={rel.mean():.2e}")
