# CEVG-RTNet: A Real-Time Architecture for Robust Forest Fire Smoke Detection in Complex Environments

This repository contains the official implementation of the detection pipeline described in our paper  
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2.1-orange)](https://pytorch.org/get-started/previous-versions/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🖧 Overall Architecture
![Architecture](https://github.com/CNNanmuzi/CEVG-RTNet/blob/main/CEVG-RTNet.png)

---

## ✨ Highlights

- Propose CEVG-RTNet for high-accuracy, real-time smoke detection on low-power devices.  
- Introduce SCPP-Conv to boost detection in low-contrast, transparent smoke scenes.  
- Use HRFA for multi-scale fusion and alignment of local details and global context.
- Design DRFE to enhance adaptability via recursive and cross-channel attention.
- Propose PolyIoU loss modeling shape, size, position via dynamic weights for accuracy.

---

## 📦 Requirements

**Tested environment：**
- Python 3.10  
- PyTorch 2.2.1  
- CUDA 12.1  

**Installation：**
```bash
git clone https://github.com/CNNanmuzi/CEVG-RTNet.git
cd CEVG-RTNet
pip install -r requirements.txt
```

---

## 📂 Dataset

The datasets used in this study consist of **public datasets** and **private datasets**.

### 📖 Public Datasets
- **SWFU-MTD**  
  🔗 [https://github.com/vinchole/zzzccc](https://github.com/vinchole/zzzccc)  
- **D-Fire**  
  🔗 [https://github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)  

### 🔒 Private Dataset
The private dataset contains unpublished experimental data and critical samples required for subsequent research.  
To avoid affecting ongoing work, it is not publicly available at this time.  
Once the paper is officially accepted, researchers with reasonable academic needs may contact the corresponding author to request access.

---

## 🚀 Experimental Run

### Training  
```bash
python train.py --cfg YOLO11-CRNet.yaml --data firesmoke.yaml --epochs 150 --imgsz 640 --batch 16 --device 0 --workers 8
```

### Validation
```bash
python val.py --weights best.pt --data firesmoke.yaml --imgsz 640 --batch 1 --device 0
```

### Inference
```bash
python predict.py --weights best.pt --source ./test/images/ --imgsz 640 --conf 0.25 --device 0
```
---

## 📜 Citation

If you use this code in your research, please cite:  
```bibtex
@inproceedings{CEVG-RTNet2025,
  title={CEVG-RTNet: A real-time architecture for robust forest fire smoke detection in complex environments},
  author={Wang, Jun and Yan, Chunman},
  booktitle={Neural Networks},
  year={2025}
}
```
