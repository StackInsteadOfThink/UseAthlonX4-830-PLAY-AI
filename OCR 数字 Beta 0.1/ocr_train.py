# -*- coding: utf-8 -*-
"""MNIST 手写数字 OCR 分类器（纯 numpy，速龙可跑，无需显卡）
784 -> 128 -> 64 -> 10，ReLU + softmax + cross-entropy，Adam
输出: 训练曲线 + 测试准确率 + 识别效果预览图
"""
import numpy as np, os, time
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))

def load():
    x = np.load(os.path.join(BASE, "train_images.npy")).reshape(-1, 784).astype(np.float32)   # 已归一化 0..1
    y = np.load(os.path.join(BASE, "train_labels.npy"))
    xt = np.load(os.path.join(BASE, "test_images.npy")).reshape(-1, 784).astype(np.float32)   # 已归一化 0..1
    yt = np.load(os.path.join(BASE, "test_labels.npy"))
    return x, y, xt, yt

def train(x, y, xt, yt, epochs=15, B=256, lr=1e-3):
    H1, H2 = 128, 64
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((H1, 784)) * np.sqrt(2/784)).astype(np.float32); b1 = np.zeros(H1, np.float32)
    W2 = (rng.standard_normal((H2, H1)) * np.sqrt(2/H1)).astype(np.float32); b2 = np.zeros(H2, np.float32)
    W3 = (rng.standard_normal((10, H2)) * np.sqrt(2/H2)).astype(np.float32); b3 = np.zeros(10, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    tt = 0
    def step(gs):
        nonlocal tt
        tt += 1
        for i, (p, g) in enumerate(zip(params, gs)):
            m[i] = 0.9*m[i] + 0.1*g; v[i] = 0.999*v[i] + 0.001*g*g
            mh = m[i]/(1-0.9**tt); vh = v[i]/(1-0.999**tt)
            p -= lr*mh/(np.sqrt(vh)+1e-8)
    N = x.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(N); tot = 0.0; cnt = 0; t0 = time.time()
        for i in range(0, N, B):
            idx = perm[i:i+B]; xb = x[idx]; yb = y[idx]
            a1 = np.maximum(xb @ W1.T + b1, 0)
            a2 = np.maximum(a1 @ W2.T + b2, 0)
            logits = a2 @ W3.T + b3
            e = np.exp(logits - logits.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
            yoh = np.zeros_like(pr); yoh[np.arange(xb.shape[0]), yb] = 1
            loss = -np.mean(np.sum(yoh * np.log(pr + 1e-9), 1))
            dlogits = (pr - yoh) / xb.shape[0]
            dW3 = dlogits.T @ a2; db3 = dlogits.sum(0)
            da2 = dlogits @ W3; dh2 = da2 * (a2 > 0)
            dW2 = dh2.T @ a1; db2 = dh2.sum(0)
            da1 = dh2 @ W2; dh1 = da1 * (a1 > 0)
            dW1 = dh1.T @ xb; db1 = dh1.sum(0)
            step([dW1, db1, dW2, db2, dW3, db3])
            tot += loss; cnt += 1
        a1 = np.maximum(xt @ W1.T + b1, 0); a2 = np.maximum(a1 @ W2.T + b2, 0)
        lg = a2 @ W3.T + b3
        acc = np.mean(np.argmax(lg, 1) == yt)
        print("epoch %d loss=%.4f test_acc=%.2f%% time=%.1fs" % (ep+1, tot/cnt, acc*100, time.time()-t0))
    return params

def preview(params, xt, yt, outpath):
    W1, b1, W2, b2, W3, b3 = params
    a1 = np.maximum(xt @ W1.T + b1, 0); a2 = np.maximum(a1 @ W2.T + b2, 0)
    pred = np.argmax(a2 @ W3.T + b3, 1)
    # 选 20 个测试图，画成 5x4 网格（含预测 vs 真值）
    rng = np.random.default_rng(7); pick = rng.choice(len(xt), 20, replace=False)
    cell = 56
    img = Image.new("RGB", (cell*5, cell*4), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for k, idx in enumerate(pick):
        r, c = k // 5, k % 5
        sub = ((1.0 - xt[idx].reshape(28, 28)) * 255.0).astype(np.uint8)   # 反色：白底黑字
        im = Image.fromarray(sub, "L").resize((cell-12, cell-12))
        img.paste(im.convert("RGB"), (c*cell+6, r*cell+6))
        d.text((c*cell+6, r*cell+cell-16), "p%d t%d" % (pred[idx], yt[idx]), fill=(0,0,0))
    img.save(outpath)
    acc = np.mean(pred == yt)
    print("预览图:", outpath, " 全测试集准确率: %.2f%%" % (acc*100))

if __name__ == "__main__":
    x, y, xt, yt = load()
    print("样本:", x.shape, "测试:", xt.shape)
    params = train(x, y, xt, yt, epochs=15)
    W1, b1, W2, b2, W3, b3 = params
    np.savez(os.path.join(BASE, "ocr_model.npz"), W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print("已保存模型权重:", os.path.join(BASE, "ocr_model.npz"))
    preview(params, xt, yt, os.path.join(BASE, "ocr_preview.png"))
