# Real_Time_Segmentation

A flexible and efficient PyTorch-based codebase for real-time semantic segmentation, supporting multiple models (including DeepLabV2 and BiSeNet) and real-world datasets like Cityscapes and GTA5. This repository is designed for research and practical applications in real-time scene understanding, with support for domain adaptation, augmentation, and advanced evaluation metrics.

---

## Features

- **Model Support**: Train and evaluate DeepLabV2 and BiSeNet (ResNet-18 backbone) for semantic segmentation.
- **Real-Time Focus**: Optimized for fast inference and efficient training, suitable for deployment and research.
- **Domain Adaptation**: Includes adversarial training scripts for adaptation between synthetic (GTA5) and real (Cityscapes) domains.
- **Flexible Data Handling**: Easily switch between Cityscapes and GTA5 datasets, with provided download and label mapping links.
- **Augmentation**: Optional advanced augmentations (horizontal flip, color jitter, Gaussian blur).
- **Performance Evaluation**: Scripts to compute FLOPs, parameter counts, latency, FPS, and detailed class-wise IoU metrics.
- **Checkpointing**: Robust checkpointing and resume support for all training scripts.

---

## Getting Started

### 1. Installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Datasets

- **Cityscapes**: [Download here](https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing)
- **GTA5**: [Download here](https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing)

Extract datasets and update the dataset paths in the scripts or use the provided defaults.

#### GTA5 to Cityscapes Label Mapping

For converting GTA5 labels to Cityscapes format, refer to [this script](https://github.com/sarrrrry/PyTorchDL_GTA5/blob/master/pytorchdl_gta5/labels.py).

---

## Model Initialization

- **DeepLab pretrained weights**: [Download here](https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing)
- Place the pretrained weights as needed and update model loading paths in your training scripts.

---

## Usage

### Training

There are several entry points under `train/`, including:
- `train_bisenet_city.py`: Train BiSeNet on Cityscapes.
- `train_deeplabv2.py`: Train DeepLabV2 on Cityscapes.
- `train_gta5.py` and `train_gta5_aug.py`: Train BiSeNet on GTA5, with or without augmentations.
- `train_adapt_adversarial.py` and `train_adapt_adversarial_focal.py`: Domain adaptation from GTA5 to Cityscapes using adversarial training (optionally with Focal Loss).

Example command:
```bash
python train/train_bisenet_city.py --data_root /path/to/Cityscapes --checkpoint_dir ./checkpoints --epochs 50 --batch_size 8
```
Adjust arguments as needed for your setup.

### Evaluation

Evaluate models using:
```bash
python val/evaluate.py --data_root /path/to/Cityscapes --model_type bisenet_resnet18 --checkpoint /path/to/checkpoint.pth.tar
```

---

## FLOPs and Parameters

Install `fvcore` for computing FLOPs:
```bash
pip install -U fvcore
```

Example to compute FLOPs:
```python
from fvcore.nn import FlopCountAnalysis, flop_count_table
# Initialize your model
image = torch.zeros((3, height, width))
flops = FlopCountAnalysis(model, image)
print(flop_count_table(flops))
```
See [fvcore docs](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) for details.

---

## Latency & FPS

Evaluation script reports latency and FPS using torch and CUDA synchronization for accurate measurement.

---

## Requirements

See `requirements.txt` for full details, including:
- torch, torchvision
- numpy, Pillow, tqdm
- fvcore (for FLOPs/parameters)
- (Optional) pyyaml, matplotlib, opencv-python

---

## License

MIT License

Copyright (c) 2025 Shohruh127

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgements

- Based on public implementations and research in semantic segmentation.
- GTA5 and Cityscapes datasets.
- [fvcore](https://github.com/facebookresearch/fvcore) for model analysis utilities.

---

## Contact

For questions or contributions, open an issue or contact [Shohruh127](https://github.com/Shohruh127).
