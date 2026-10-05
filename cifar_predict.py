# -*- coding: utf-8 -*-
"""CIFAR-10 图像识别入口（v2 模型）：输入任意图片 -> 输出这是什么
用法: python cifar_predict.py <图片路径>
"""
import numpy as np, os, sys
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]

def load_model():
    z = np.load(os.path.join(BASE, "cifar_model_v2.npz"))
    return [z['W1'],z['b1'],z['W2'],z['b2'],z['W3'],z['b3'],z['W4'],z['b4']]

def im2col(x, kh, kw, pad=0):
    N,C,H,W = x.shape; Hp=H+2*pad-kh+1; Wp=W+2*pad-kw+1
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    cols = np.empty((N,kh,kw,C,Hp,Wp), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:,i,j,:,:,:] = xp[:,:,i:i+Hp,j:j+Wp]
    return cols.transpose(0,4,5,3,1,2).reshape(N*Hp*Wp, C*kh*kw)

def forward(x, params):
    W1,b1,W2,b2,W3,b3,W4,b4 = params; B=x.shape[0]
    cols1 = im2col(x,3,3,1)
    h1 = (cols1@W1.reshape(27,16)).reshape(B,32,32,16).transpose(0,3,1,2)+b1.reshape(1,16,1,1)
    a1 = np.maximum(h1,0)
    p1 = a1.reshape(B,16,16,2,16,2).mean(axis=(3,5))
    cols2 = im2col(p1,3,3,1)
    h2 = (cols2@W2.reshape(144,32)).reshape(B,16,16,32).transpose(0,3,1,2)+b2.reshape(1,32,1,1)
    a2 = np.maximum(h2,0)
    p2 = a2.reshape(B,32,8,2,8,2).mean(axis=(3,5))
    f = p2.reshape(B,2048)
    a3 = np.maximum(f@W3+b3,0)
    return a3@W4+b4

def preprocess(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((32,32), Image.LANCZOS)
    return np.array(im, dtype=np.float32).transpose(2,0,1).reshape(1,3,32,32)/255.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python cifar_predict.py <图片路径>"); sys.exit(1)
    img = preprocess(sys.argv[1])
    logits = forward(img, load_model())
    p = int(np.argmax(logits,1)[0])
    pr = np.exp(logits-logits.max(1,keepdims=True)); pr = (pr/pr.sum(1,keepdims=True))[0]
    print("图片:", sys.argv[1])
    print("识别为: %s  (置信度 %.1f%%)" % (LABELS[p], pr[p]*100))
    print("--- 各类别概率 ---")
    for i,(l,v) in enumerate(zip(LABELS, pr)):
        print("  %s: %.1f%%" % (l, v*100))
