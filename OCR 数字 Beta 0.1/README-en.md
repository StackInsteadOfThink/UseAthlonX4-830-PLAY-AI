# OCR Digit Recognition (Handwritten Digits)

A **handwritten-digit recognition** classifier trained on an Athlon X4 830 (pure CPU), using the classic MNIST dataset (60k 28×28 handwritten digit images). It can tell which of 0–9 a digit image is.

## What each file does

| File | Purpose |
|---|---|
| `download_mnist.py` | Downloads the MNIST training data (needs internet), generates npy files |
| `ocr_train.py` | Trains the model, saves the result as `ocr_model.npz` |
| `ocr_model.npz` | The trained model weights |
| `ocr_preview.png` | A preview image of the training result |
| `train_images.npy` / `train_labels.npy` | Training images and labels |
| `test_images.npy` / `test_labels.npy` | Test images and labels |
| `list.txt` | Notes file |

## How to use

1. First run `download_mnist.py` to download the data (skip if the npy data is already included).
2. Run `ocr_train.py` to train, generating `ocr_model.npz`.
3. Use `ocr_model.npz` for inference to recognize handwritten digits.

(Note: this project has no GUI demo yet — the main output is the training pipeline + weight file.)

## Notes

- `train_images.npy` is ~180 MB and is public dataset (MNIST); if deleted, re-download it.
- Requires Python 3.12 + numpy.
