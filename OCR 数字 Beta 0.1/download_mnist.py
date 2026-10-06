# -*- coding: utf-8 -*-
"""下载 MNIST 并转为 numpy .npy（纯 numpy，无需 torch）"""
import gzip, os, urllib.request, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

def dl(name, url):
    path = os.path.join(BASE, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("已存在", name, "跳过下载")
        return path
    print("下载", name, "...")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, path)
            print("  完成", name, os.path.getsize(path), "bytes")
            return path
        except Exception as e:
            print("  第%d次失败: %s" % (attempt+1, e))
    raise RuntimeError("下载失败: " + name)

def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)

if __name__ == "__main__":
    for name, url in URLS.items():
        dl(name, url)
    x_train = read_idx_images(os.path.join(BASE, "train-images-idx3-ubyte.gz"))
    y_train = read_idx_labels(os.path.join(BASE, "train-labels-idx1-ubyte.gz"))
    x_test  = read_idx_images(os.path.join(BASE, "t10k-images-idx3-ubyte.gz"))
    y_test  = read_idx_labels(os.path.join(BASE, "t10k-labels-idx1-ubyte.gz"))
    np.save(os.path.join(BASE, "train_images.npy"), x_train)
    np.save(os.path.join(BASE, "train_labels.npy"), y_train)
    np.save(os.path.join(BASE, "test_images.npy"), x_test)
    np.save(os.path.join(BASE, "test_labels.npy"), y_test)
    print("OK shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
