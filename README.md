# CEVG-RTNet: A Real-Time Architecture for Robust Forest Fire Smoke Detection in Complex Environments

This repository contains the official implementation of the detection pipeline described in our paper  
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2.1-orange)](https://pytorch.org/get-started/previous-versions/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)


## 📦 Installation

**Tested environment：**
- Python 3.10
- PyTorch 2.2.1
- CUDA 12.1

**Steps：**
```bash
git clone https://github.com/CNNanmuzi/CEVG-RTNet/.git
cd <CEVG-RTNet>
pip install -r requirements.txt
```

---

## 📂 Data Preparation / 数据准备

1. Download the dataset from [<Dataset Link>]  
   从 [<数据集下载链接>] 下载数据集  
2. Extract it into the `data/` directory, or update the path in `configs/config.yaml`  
   解压到 `data/` 目录，或在 `configs/config.yaml` 中修改路径

**Expected folder structure / 期望的数据目录结构：**
```
data/
    dataset_folder/
        images/
        labels/
```

---

## 🚀 Running Experiments / 运行实验

### Run the default experiment (reproduces Table 2 from the paper)  
运行默认实验（复现论文 Table 2 结果）：
```bash
bash scripts/run_experiment.sh
```

### Run with a custom config / 使用自定义配置文件运行：
```bash
python src/main.py --config configs/config.yaml
```

---

## 📊 Expected Results / 期望结果

If everything is set up correctly, you should see results similar to those in the paper.  
如果环境配置正确，结果应与论文中报告的指标接近。

**Example output / 示例输出：**
```
mAP: 78.5
Precision: 82.1
Recall: 77.4
```

---

## ⚙ Configuration / 配置说明

Modify `configs/config.yaml` to change / 修改 `configs/config.yaml` 可以调整：
- Dataset path / 数据集路径
- Model architecture / 模型结构
- Training hyperparameters / 训练超参数

---

## 📜 Citation / 引用

If you use this code in your research, please cite:  
如果您在研究中使用了本代码，请引用：

```bibtex
@inproceedings{<yourname2025detection>,
  title={<Your Paper Title>},
  author={<Author1> and <Author2> and <Author3>},
  booktitle={<Conference/Journal>},
  year={2025}
}
```

---

## 📧 Contact / 联系方式

For questions, please contact / 如有问题，请联系：
- <your.email@example.com>
