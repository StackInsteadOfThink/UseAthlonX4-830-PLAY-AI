# -*- coding: utf-8 -*-
"""CIFAR-10 强化训练 v2（速龙纯 numpy）
改进: 训练时水平翻转数据增强 + 更多 epoch
网络: conv3->16 -> relu -> pool -> conv16->32 -> relu -> pool -> fc2048->128 -> fc10
数据与模型都放在脚本所在目录（D:\训练 picture）
"""
import numpy as np, os, pickle, time
BASE = os.path.dirname(os.path.abspath(__file__))

def load_cifar(data_dir):
    xs, ys = [], []
    for b in range(1, 6):
        with open(os.path.join(data_dir, "data_batch_%d" % b), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32)); ys += d[b"labels"]
    X = np.concatenate(xs).astype(np.float32) / 255.0
    y = np.array(ys, dtype=np.int64)
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    Xt = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    yt = np.array(d[b"labels"], dtype=np.int64)
    return X, y, Xt, yt

def im2col(x, kh, kw, pad=0):
    N, C, H, W = x.shape
    Hp = H + 2*pad - kh + 1; Wp = W + 2*pad - kw + 1
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    cols = np.empty((N, kh, kw, C, Hp, Wp), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, i, j, :, :, :] = xp[:, :, i:i+Hp, j:j+Wp]
    return cols.transpose(0, 4, 5, 3, 1, 2).reshape(N*Hp*Wp, C*kh*kw)

def col2im(dcols, xshape, kh, kw, pad=0):
    N, C, H, W = xshape
    Hp = H + 2*pad - kh + 1; Wp = W + 2*pad - kw + 1
    dcols = dcols.reshape(N, Hp, Wp, C, kh, kw)
    dxp = np.zeros((N, C, H+2*pad, W+2*pad), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i:i+Hp, j:j+Wp] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    if pad: dxp = dxp[:, :, pad:-pad, pad:-pad]
    return dxp

def forward(x, params):
    W1,b1,W2,b2,W3,b3,W4,b4 = params; B = x.shape[0]
    cols1 = im2col(x, 3,3,1)
    h1 = (cols1 @ W1.reshape(27,16)).reshape(B,32,32,16).transpose(0,3,1,2) + b1.reshape(1,16,1,1)
    a1 = np.maximum(h1, 0)
    p1 = a1.reshape(B,16,16,2,16,2).mean(axis=(3,5))
    cols2 = im2col(p1, 3,3,1)
    h2 = (cols2 @ W2.reshape(144,32)).reshape(B,16,16,32).transpose(0,3,1,2) + b2.reshape(1,32,1,1)
    a2 = np.maximum(h2, 0)
    p2 = a2.reshape(B,32,8,2,8,2).mean(axis=(3,5))
    f = p2.reshape(B, 2048)
    a3 = np.maximum(f @ W3 + b3, 0)
    logits = a3 @ W4 + b4
    return logits, (x, cols1, h1, a1, p1, cols2, h2, a2, p2, f, a3)

def backward(params, cache, logits, y, B):
    W1,b1,W2,b2,W3,b3,W4,b4 = params
    x, cols1, h1, a1, p1, cols2, h2, a2, p2, f, a3 = cache
    e = np.exp(logits - logits.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
    yoh = np.zeros_like(pr); yoh[np.arange(B), y] = 1
    dlog = (pr - yoh) / B
    dW4 = a3.T @ dlog; db4 = dlog.sum(0)
    da3 = dlog @ W4.T; dh3 = da3 * (a3 > 0)
    dW3 = f.T @ dh3; db3 = dh3.sum(0)
    df = dh3 @ W3.T
    da2 = df.reshape(B,32,8,1,8,1).repeat(2, axis=3).repeat(2, axis=5).reshape(B,32,16,16) / 4.0
    dh2 = da2 * (a2 > 0)
    dcols2 = dh2.transpose(0,2,3,1).reshape(-1,32)
    dW2 = cols2.T @ dcols2; db2 = dcols2.sum(0)
    dp1 = col2im(dcols2 @ W2.reshape(144,32).T, p1.shape, 3,3,1)
    da1 = dp1.reshape(B,16,16,1,16,1).repeat(2, axis=3).repeat(2, axis=5).reshape(B,16,32,32) / 4.0
    dh1 = da1 * (a1 > 0)
    dcols1 = dh1.transpose(0,2,3,1).reshape(-1,16)
    dW1 = cols1.T @ dcols1; db1 = dcols1.sum(0)
    return [dW1, db1, dW2, db2, dW3, db3, dW4, db4]

def train(X, y, Xt, yt, epochs=20, B=64, lr=1e-3):
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((27,16))*np.sqrt(2/27)).astype(np.float32); b1=np.zeros(16,np.float32)
    W2 = (rng.standard_normal((144,32))*np.sqrt(2/144)).astype(np.float32); b2=np.zeros(32,np.float32)
    W3 = (rng.standard_normal((2048,128))*np.sqrt(2/2048)).astype(np.float32); b3=np.zeros(128,np.float32)
    W4 = (rng.standard_normal((128,10))*np.sqrt(2/128)).astype(np.float32); b4=np.zeros(10,np.float32)
    params = [W1,b1,W2,b2,W3,b3,W4,b4]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    tt = 0
    def step(gs):
        nonlocal tt
        tt += 1
        for i,(p,g) in enumerate(zip(params,gs)):
            m[i]=0.9*m[i]+0.1*g; v[i]=0.999*v[i]+0.001*g*g
            p -= lr*(m[i]/(1-0.9**tt))/(np.sqrt(v[i]/(1-0.999**tt))+1e-8)
    N = len(X)
    best = 0.0
    for ep in range(epochs):
        perm = rng.permutation(N); tot=0; cnt=0; t0=time.time()
        for i in range(0, N, B):
            idx = perm[i:i+B]; xb = X[idx].copy(); yb = y[idx]
            # 水平翻转增强（0.5 概率）
            if rng.random() < 0.5: xb = xb[:, :, :, ::-1]
            logits, cache = forward(xb, params)
            lsm = logits - logits.max(1, keepdims=True)
            loss = (lsm[np.arange(len(xb)), yb] - np.log(np.exp(lsm).sum(1))).mean()
            gs = backward(params, cache, logits, yb, len(xb))
            step(gs)
            tot += -float(loss); cnt += 1
        acc = eval_acc(Xt, yt, params, 256)
        if acc > best:
            best = acc
            np.savez(os.path.join(BASE, "cifar_model_v2.npz"),
                     W1=params[0],b1=params[1],W2=params[2],b2=params[3],
                     W3=params[4],b3=params[5],W4=params[6],b4=params[7])
        print("epoch %d loss=%.4f test_acc=%.2f%% time=%.1fs" % (ep+1, tot/cnt, acc*100, time.time()-t0), flush=True)
    print("最佳测试准确率: %.2f%%" % (best*100))
    return params

def eval_acc(X, y, params, B=256):
    preds = []
    for i in range(0, len(X), B):
        lg, _ = forward(X[i:i+B], params)
        preds.append(np.argmax(lg, 1))
    return np.mean(np.concatenate(preds) == y)

if __name__ == "__main__":
    import sys
    data_dir = os.path.join(BASE, "cifar-10-batches-py")
    if not os.path.isdir(data_dir):
        print("找不到数据:", data_dir); sys.exit(1)
    print("加载 CIFAR-10 ...")
    X, y, Xt, yt = load_cifar(data_dir)
    print("训练:", X.shape, "测试:", Xt.shape)
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("训练 %d epoch（含水平翻转增强）..." % ep)
    params = train(X, y, Xt, yt, epochs=ep)
    np.savez(os.path.join(BASE, "cifar_model_v2.npz"),
             W1=params[0],b1=params[1],W2=params[2],b2=params[3],
             W3=params[4],b3=params[5],W4=params[6],b4=params[7])
    print("已保存:", os.path.join(BASE, "cifar_model_v2.npz"))
