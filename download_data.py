# -*- coding: utf-8 -*-
"""下载并解压 CIFAR-10 数据集（优先国内镜像，失败回退官方源）
用法: python download_data.py
"""
import os, sys, urllib.request, tarfile

BASE = os.path.dirname(os.path.abspath(__file__))
URLS = [
    "http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
NAME = "cifar-10-python.tar.gz"

def download(url):
    print("下载:", url)
    urllib.request.urlretrieve(url, os.path.join(BASE, NAME))
    return os.path.getsize(os.path.join(BASE, NAME))

if __name__ == "__main__":
    if os.path.isdir(os.path.join(BASE, "cifar-10-batches-py")):
        print("数据已存在，跳过下载。"); sys.exit(0)
    ok = False
    for u in URLS:
        try:
            size = download(u)
            print("下载完成:", size, "字节")
            ok = True; break
        except Exception as e:
            print("失败:", e)
    if not ok:
        print("所有源均失败，请手动下载 CIFAR-10 并解压到本目录"); sys.exit(1)
    with tarfile.open(os.path.join(BASE, NAME), "r:gz") as t:
        t.extractall(BASE)
    print("解压完成: cifar-10-batches-py/")
