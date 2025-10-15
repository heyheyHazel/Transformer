# Transformer
PyTorch implementation of Transformer: Attention is all you need.

![GitHub last commit](https://img.shields.io/github/last-commit/heyheyHazel/Transformer)
![GitHub repo size](https://img.shields.io/github/repo-size/heyheyHazel/Transformer)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

## 🎯 项目简介

从零开始实现Transformer架构的完整代码库，包含基础Transformer组件、完整模型实现以及实际应用案例。本项目旨在深入理解Transformer的工作原理，并提供可复用的实现代码。


## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.6.0
- **核心语言**: Python 3.11.13
- **数据处理**: NumPy｜Pandas
- **可视化**: Matplotlib｜Seaborn

## 📁 项目目录

### 🔧 Transformer完整代码实现

- [🧩 Transformer完整代码注释](transformer.ipynb)
  - [《Attention Is All You Need》]([models/vanilla_transformer.py](http://arxiv.org/abs/1706.03762)) - 原始论文阅读（可参考[Transformer论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=b30b07507c510812227479ae70dadeba)）




## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/heyheyHazel/Transformer.git
cd Transformer

# 创建虚拟环境并安装基础的包(conda)
conda create -n pytorch_env python=3.9 -y
conda activate pytorch_env
conda install pytorch pandas numpy matplotlib ipykernel  -y
