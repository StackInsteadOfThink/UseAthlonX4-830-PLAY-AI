# -*- coding: utf-8 -*-
"""
MiniChat Beta 0.1 —— 纯 numpy 训练 300 万参数字符级 transformer
速龙 X4 830 纯 CPU 硬跑。支持断点续训、梯度检查(--check)。
"""
import numpy as np, os, math, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1337)

CFG = {
    "n_embd": 256, "n_head": 4, "n_layer": 4, "block_size": 128,
    "batch_size": 4, "lr": 3e-3, "weight_decay": 0.1,
    "max_iters": 15000, "eval_interval": 200, "log_interval": 20,
    "ckpt_interval": 1000, "grad_clip": 1.0,
}
CKPT = os.path.join(HERE, "minichat.npz")
LOG   = os.path.join(HERE, "train_log.txt")

def load_data():
    d = np.load(os.path.join(HERE, "data.npz"))
    return d["train"], d["val"], int(d["vocab_size"]), str(d["chars"])

train_data, val_data, vocab_size, chars = load_data()

def get_batch(split):
    data = train_data if split == "train" else val_data
    n = len(data); T = CFG["block_size"]; B = CFG["batch_size"]
    ix = np.random.randint(0, n - T - 1, B)
    x = np.stack([data[i:i+T] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+T+1] for i in ix]).astype(np.int64)
    return x, y

def init_params():
    P = {}; n = CFG["n_embd"]; sc = 1/math.sqrt(n); nl = CFG["n_layer"]
    def m(r, c): return (np.random.randn(r, c)*sc).astype(np.float32)
    P["wte"] = (np.random.randn(vocab_size, n)*0.02).astype(np.float32)
    P["wpe"] = (np.random.randn(CFG["block_size"], n)*0.02).astype(np.float32)
    for l in range(nl):
        P[f"attn_{l}_wq"] = m(n, n); P[f"attn_{l}_wk"] = m(n, n)
        P[f"attn_{l}_wv"] = m(n, n); P[f"attn_{l}_wo"] = m(n, n)
        P[f"mlp_{l}_w1"] = m(4*n, n); P[f"mlp_{l}_w2"] = m(n, 4*n)
        P[f"mlp_{l}_b1"] = np.zeros(4*n, np.float32); P[f"mlp_{l}_b2"] = np.zeros(n, np.float32)
        P[f"ln1_{l}_g"] = np.ones(n, np.float32);  P[f"ln1_{l}_b"] = np.zeros(n, np.float32)
        P[f"ln2_{l}_g"] = np.ones(n, np.float32);  P[f"ln2_{l}_b"] = np.zeros(n, np.float32)
    P["lnf_g"] = np.ones(n, np.float32); P["lnf_b"] = np.zeros(n, np.float32)
    return P

def layernorm_fwd(x, g, b):
    mean = x.mean(-1, keepdims=True); std = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    hn = (x - mean) / std
    return hn * g + b, (mean, std, hn)

def gelu(x):
    c = math.sqrt(2/math.pi)
    return 0.5*x*(1+np.tanh(c*(x + 0.044715*x**3)))

def gelu_grad(x):
    c = math.sqrt(2/math.pi); u = x + 0.044715*x**3; t = np.tanh(c*u)
    return 0.5*(1+t) + 0.5*x*(1-t*t)*c*(1+0.134145*x**2)

def softmax(x, axis=-1):
    m = x.max(axis, keepdims=True); e = np.exp(x-m); return e/e.sum(axis, keepdims=True)

def forward(P, x):
    B, T = x.shape; n = CFG["n_embd"]; hs = n // CFG["n_head"]; nl = CFG["n_layer"]
    C = {}
    emb = P["wte"][x]
    h = emb + P["wpe"][:T][None, :, :]
    C["emb"] = emb; C["x"] = x
    mask = np.triu(np.ones((T, T), bool), 1)
    for l in range(nl):
        h1, l1 = layernorm_fwd(h, P[f"ln1_{l}_g"], P[f"ln1_{l}_b"])
        Q = h1 @ P[f"attn_{l}_wq"].T; K = h1 @ P[f"attn_{l}_wk"].T; V = h1 @ P[f"attn_{l}_wv"].T
        Q = Q.reshape(B, T, CFG["n_head"], hs).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, CFG["n_head"], hs).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, CFG["n_head"], hs).transpose(0, 2, 1, 3)
        att = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(hs)
        att = np.where(mask[None, None, :, :], -1e9, att)
        asm = softmax(att, -1)
        ctx = asm @ V
        ctxc = ctx.transpose(0, 2, 1, 3).reshape(B, T, n)
        aout = ctxc @ P[f"attn_{l}_wo"].T
        h = h + aout
        h2, l2 = layernorm_fwd(h, P[f"ln2_{l}_g"], P[f"ln2_{l}_b"])
        m1v = h2 @ P[f"mlp_{l}_w1"].T + P[f"mlp_{l}_b1"]
        gv = gelu(m1v)
        m2v = gv @ P[f"mlp_{l}_w2"].T + P[f"mlp_{l}_b2"]
        h = h + m2v
        C[f"attn_{l}_h1"] = h1; C[f"attn_{l}_ln1"] = l1
        C[f"attn_{l}_Q"] = Q; C[f"attn_{l}_K"] = K; C[f"attn_{l}_V"] = V
        C[f"attn_{l}_asm"] = asm; C[f"attn_{l}_ctxc"] = ctxc
        C[f"mlp_{l}_ln2"] = l2; C[f"mlp_{l}_h2"] = h2
        C[f"mlp_{l}_m1"] = m1v; C[f"mlp_{l}_gv"] = gv
    hf, lf = layernorm_fwd(h, P["lnf_g"], P["lnf_b"])
    logits = hf @ P["wte"].T
    C["lnf"] = lf; C["hf"] = hf
    return logits, C

def layernorm_bwd(dy, std, hn, g, dg, db):
    d_hn = dy * g
    dg[:] = (dy * hn).sum(axis=(0, 1)); db[:] = dy.sum(axis=(0, 1))
    return (d_hn - d_hn.mean(-1, keepdims=True) - hn * (d_hn * hn).mean(-1, keepdims=True)) / std

def backward(P, C, dlogits):
    n = CFG["n_embd"]; hs = n // CFG["n_head"]; nl = CFG["n_layer"]; B, T = dlogits.shape[:2]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    d_hf = dlogits @ P["wte"]
    G["wte"] += (dlogits.transpose(0, 2, 1) @ C["hf"]).sum(0)
    d_h = layernorm_bwd(d_hf, C["lnf"][1], C["lnf"][2], P["lnf_g"], G["lnf_g"], G["lnf_b"])
    mask = np.triu(np.ones((T, T), bool), 1)[None, None, :, :]
    for l in reversed(range(nl)):
        d_m2 = d_h
        G[f"mlp_{l}_b2"] += d_m2.sum((0, 1))
        G[f"mlp_{l}_w2"] += (d_m2.transpose(0, 2, 1) @ C[f"mlp_{l}_gv"]).sum(0)
        d_gv = d_m2 @ P[f"mlp_{l}_w2"]
        d_m1 = d_gv * gelu_grad(C[f"mlp_{l}_m1"])
        G[f"mlp_{l}_b1"] += d_m1.sum((0, 1))
        G[f"mlp_{l}_w1"] += (d_m1.transpose(0, 2, 1) @ C[f"mlp_{l}_h2"]).sum(0)
        d_h2 = d_m1 @ P[f"mlp_{l}_w1"]
        d_h_mid = layernorm_bwd(d_h2, C[f"mlp_{l}_ln2"][1], C[f"mlp_{l}_ln2"][2],
                                P[f"ln2_{l}_g"], G[f"ln2_{l}_g"], G[f"ln2_{l}_b"])
        d_h_mid += d_h
        d_aout = d_h_mid
        d_ctxc = d_aout @ P[f"attn_{l}_wo"]
        G[f"attn_{l}_wo"] += (d_aout.transpose(0, 2, 1) @ C[f"attn_{l}_ctxc"]).sum(0)
        dctx = d_ctxc.reshape(B, T, CFG["n_head"], hs).transpose(0, 2, 1, 3)
        d_asm = dctx @ C[f"attn_{l}_V"].transpose(0, 1, 3, 2)
        dV = C[f"attn_{l}_asm"].transpose(0, 1, 3, 2) @ dctx
        datt = C[f"attn_{l}_asm"] * (d_asm - (d_asm * C[f"attn_{l}_asm"]).sum(-1, keepdims=True))
        datt = np.where(mask, 0, datt)
        dQ = datt @ C[f"attn_{l}_K"] / math.sqrt(hs)
        dK = datt.transpose(0, 1, 3, 2) @ C[f"attn_{l}_Q"] / math.sqrt(hs)
        dQf = dQ.transpose(0, 2, 1, 3).reshape(B, T, n)
        dKf = dK.transpose(0, 2, 1, 3).reshape(B, T, n)
        dVf = dV.transpose(0, 2, 1, 3).reshape(B, T, n)
        G[f"attn_{l}_wq"] += (dQf.transpose(0, 2, 1) @ C[f"attn_{l}_h1"]).sum(0)
        G[f"attn_{l}_wk"] += (dKf.transpose(0, 2, 1) @ C[f"attn_{l}_h1"]).sum(0)
        G[f"attn_{l}_wv"] += (dVf.transpose(0, 2, 1) @ C[f"attn_{l}_h1"]).sum(0)
        d_h1 = dQf @ P[f"attn_{l}_wq"] + dKf @ P[f"attn_{l}_wk"] + dVf @ P[f"attn_{l}_wv"]
        # h_in = LN1^{-1}(h1); h_mid = h_in + aout → 残差梯度加到 LN1 反向之后
        d_h = layernorm_bwd(d_h1, C[f"attn_{l}_ln1"][1], C[f"attn_{l}_ln1"][2],
                            P[f"ln1_{l}_g"], G[f"ln1_{l}_g"], G[f"ln1_{l}_b"]) + d_h_mid
    d_emb = d_h
    G["wpe"][:T] += d_emb.sum(axis=0)
    np.add.at(G["wte"], C["x"].ravel(), d_emb.reshape(-1, n))
    return G

def cross_entropy(logits, y):
    B, T, V = logits.shape
    p = softmax(logits, -1)
    loss = -np.log(p[np.arange(B)[:, None], np.arange(T)[None, :], y] + 1e-9).mean()
    dlogits = p.copy(); dlogits[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1
    dlogits /= (B * T)
    return loss, dlogits

def param_count(P):
    return sum(int(v.size) for v in P.values())

# ---------- 梯度检查 ----------
def gradient_check():
    global CFG, train_data, val_data, vocab_size, chars
    CFG = dict(CFG, n_embd=8, n_head=2, n_layer=2, block_size=6, batch_size=2)
    vocab_size = 12
    np.random.seed(1)
    P = init_params()
    P = {k: v.astype(np.float64) for k, v in P.items()}
    x = np.random.randint(0, 12, (2, 6)).astype(np.int64)
    y = np.random.randint(0, 12, (2, 6)).astype(np.int64)
    logits, C = forward(P, x)
    loss, dlogits = cross_entropy(logits, y)
    G = backward(P, C, dlogits)
    # numerical gradient
    eps = 1e-4; keys = list(P.keys()); worst = 0.0; bad = []
    for k in keys:
        W = P[k].copy()
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            W2 = P[k].copy(); W2[i] += eps
            P2 = dict(P); P2[k] = W2
            lg, _ = forward(P2, x); l1, _ = cross_entropy(lg, y)
            W3 = P[k].copy(); W3[i] -= eps
            P3 = dict(P); P3[k] = W3
            lg2, _ = forward(P3, x); l2, _ = cross_entropy(lg2, y)
            num[i] = (l1 - l2) / (2 * eps)
        denom = np.maximum(np.abs(num), np.abs(G[k])) + 1e-9
        rel = np.abs(num - G[k]) / denom
        m = float(rel.max())
        if m > 1e-3:
            bad.append((k, m))
        worst = max(worst, m)
    print(f"梯度检查: 最大相对误差={worst:.2e}")
    if bad:
        print("超标参数:", bad)
        sys.exit(1)
    print("梯度检查通过 ✓")

# ---------- 训练 ----------
def train():
    print("参数总量: {:.2f} 万".format(param_count(init_params()) / 1e4))
    P = None; m = None; v = None; it = 0; t0 = time.time()
    if os.path.exists(CKPT):
        d = np.load(CKPT, allow_pickle=True)
        P = {k: d[k] for k in d.files if k not in ("iters",)}
        it = int(d["iters"]); print(f"从断点恢复: 第 {it} 迭代")
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
    else:
        P = init_params()
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
    logf = open(LOG, "a", encoding="utf-8")
    lr = CFG["lr"]; wd = CFG["weight_decay"]; b1 = 0.9; b2 = 0.95; eps = 1e-8
    iters = CFG["max_iters"]
    while it < iters:
        x, y = get_batch("train")
        logits, C = forward(P, x)
        loss, dlogits = cross_entropy(logits, y)
        G = backward(P, C, dlogits)
        # grad clip
        total = math.sqrt(sum((v* v).sum() for v in G.values()))
        if total > CFG["grad_clip"]:
            sc = CFG["grad_clip"] / total
            G = {k: g*sc for k, g in G.items()}
        # AdamW
        for k in P:
            g = G[k]
            if k.endswith(("_b1", "_b2", "_g", "_b")):
                g = g  # no decay on biases/ln
            else:
                g = g + wd * P[k]
            m[k] = b1*m[k] + (1-b1)*g
            v[k] = b2*v[k] + (1-b2)*(g*g)
            mh = m[k]/(1-b1**(it+1)); vh = v[k]/(1-b2**(it+1))
            P[k] -= lr * mh / (np.sqrt(vh) + eps)
        it += 1
        if it % CFG["log_interval"] == 0:
            spd = it / (time.time() - t0)
            msg = f"iter {it:6d} | loss {loss:.4f} | {spd:.3f} iter/s | 预计剩余 {(iters-it)/spd/3600:.1f} h"
            print(msg); logf.write(msg + "\n"); logf.flush()
        if it % CFG["eval_interval"] == 0:
            ev = []
            for _ in range(5):
                xe, ye = get_batch("val")
                le, _ = forward(P, xe); el, _ = cross_entropy(le, ye); ev.append(float(el))
            msg = f"[eval] iter {it} val_loss {float(np.mean(ev)):.4f}"
            print(msg); logf.write(msg + "\n"); logf.flush()
        if it % CFG["ckpt_interval"] == 0:
            save = {**{k: v for k, v in P.items()}, "iters": it}
            np.savez(CKPT, **save)
            msg = f"[ckpt] 已保存到 {CKPT} (iter {it})"
            print(msg); logf.write(msg + "\n"); logf.flush()
    # final save
    save = {**{k: v for k, v in P.items()}, "iters": it}
    np.savez(CKPT, **save)
    logf.write("[done] 训练完成\n"); logf.close()
    print("训练完成 ✓")

if __name__ == "__main__":
    if "--check" in sys.argv:
        gradient_check()
    else:
        train()
