# OCR 数字识别（手写数字）

在速龙 X4 830（纯 CPU）上训练的**手写数字识别**分类器，用经典的 MNIST 数据集（6 万张 28×28 手写数字图），能认出 0–9 是哪张图。

## 文件都是干嘛的

| 文件 | 作用 |
|---|---|
| `download_mnist.py` | 下载 MNIST 训练数据（需要联网），生成 npy 文件 |
| `ocr_train.py` | 训练模型，把结果存成 `ocr_model.npz` |
| `ocr_model.npz` | 训练好的模型权重 |
| `ocr_preview.png` | 训练出的效果预览图 |
| `train_images.npy` / `train_labels.npy` | 训练用的图片和标签数据 |
| `test_images.npy` / `test_labels.npy` | 测试用的图片和标签数据 |
| `list.txt` | 备注文件 |

## 怎么用

1. 先跑 `download_mnist.py` 下载数据（已附带 npy 数据则可跳过）。
2. 跑 `ocr_train.py` 训练，生成 `ocr_model.npz`。
3. 用 `ocr_model.npz` 做推理识别手写数字。

（注：本项目还没有图形化 demo，主力是训练 + 权重文件。）

## 注意

- `train_images.npy` 约 180MB，是公开数据集（MNIST），删了可重新下载。
- 需要 Python 3.12 + numpy。
