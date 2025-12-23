# BiLSTM with Attention for Yelp Review Classification

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Project Overview

This project implements a Bidirectional LSTM (BiLSTM) with Attention mechanism for sentiment classification on Yelp review data. The model classifies reviews into 5 star ratings (1-5 stars) using deep learning techniques.

### Features

- **BiLSTM Architecture**: Bidirectional LSTM for better context understanding
- **Attention Mechanism**: Weighted attention to focus on important words
- **Flexible Configuration**: Multiple hyperparameters configurable via YAML
- **Distributed Training**: Support for multi-GPU training with PyTorch DDP
- **Experiment Tracking**: Compare different model configurations
- **Visualization**: Automatic generation of training curves and comparison plots

### Prerequisites

- Anaconda or Miniconda
- NVIDIA GPU with CUDA support (recommended)
- Python 3.11
- PyTorch with CUDA 12.1

### Directory Structure Setup

Before starting, create the necessary directories:

```bash
# Create dataset directory
mkdir dataset

# Create saved_models directory with subdirectories
mkdir -p saved_models/figures
mkdir -p saved_models/models
```

After setup, your directory structure should look like:
```
BiLstm-with-attention/
├── dataset/                          # Dataset directory
│   ├── yelp_academic_dataset_review.json  # Training data (to be downloaded)
│   └── test.json                     # Test data (to be downloaded)
├── saved_models/                     # Model outputs directory
│   ├── figures/                      # Training plots and visualizations
│   └── models/                       # Saved model checkpoints
├── model/                            # Model architecture
│   └── Bilstm_attention.py
├── utils/                            # Utility functions
│   └── data.py
├── config.yaml                       # Configuration file
├── environment.yml                   # Conda environment file
├── train.py                          # Single-GPU training script
└── train_distributed.py              # Multi-GPU distributed training script
```

### Dataset Download

Download the Yelp Academic Dataset from the following GitHub repository:

**Repository**: [Yelp Data Challenge 2013](https://github.com/rekiksab/Yelp-Data-Challenge-2013/tree/master/yelp_challenge/yelp_phoenix_academic_dataset)

**Steps**:
1. Visit the repository: https://github.com/rekiksab/Yelp-Data-Challenge-2013
2. Navigate to `yelp_challenge/yelp_phoenix_academic_dataset/`
3. Download the following files:
   - `yelp_academic_dataset_review.json` (training data)
   - Any test data file and rename it to `test.json`
4. Place the downloaded files in the `dataset/` directory

Alternatively, you can use `wget` or `git`:

```bash
# Clone the repository
git clone https://github.com/rekiksab/Yelp-Data-Challenge-2013.git

# Copy the dataset files to your dataset directory
cp Yelp-Data-Challenge-2013/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json dataset/

# Create or copy test data
cp Yelp-Data-Challenge-2013/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json dataset/test.json
```

### Environment Setup

Create and configure the conda environment:

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate dl-exp

# Install additional dependencies if needed
pip install pyyaml
```

The `environment.yml` file includes:
- Python 3.11
- PyTorch with CUDA 12.1 support (pytorch-cuda=12.1)
- TorchVision and TorchAudio
- Pandas for data processing
- Matplotlib for visualization

### Configuration

Edit `config.yaml` to customize training parameters:

```yaml
base:
  train_path: dataset/yelp_academic_dataset_review.json
  test_path: dataset/test.json
  max_vocab_size: 50000
  max_len: 256
  min_freq: 5
  batch_size: 64
  epochs: 15
  learning_rate: 0.001
  device: cuda
  seed: 42
  embed_dim: 256
  hidden_dim: 256
  n_layers: 2
  dropout: 0.3
  use_layer_norm: true
  use_residual: true
  lr_decay: true
  num_classes: 5

experiments:
  Baseline: {}
  dropout_0.5:
    dropout: 0.5
  # Add more experimental configurations as needed
```

### Training

#### Single-GPU Training

For training on a single GPU:

```bash
# Activate the conda environment
conda activate dl-exp

# Run training
python train.py
```

The script will:
- Load and preprocess the dataset
- Build vocabulary from training data
- Run all experiments defined in `config.yaml`
- Save the best model to `saved_models/models/`
- Generate training plots in `saved_models/figures/`

#### Multi-GPU Distributed Training

For distributed training across multiple GPUs:

**Important**: Before running, modify the `world_size` parameter in `train_distributed.py` to match your number of GPUs.

```python
# In train_distributed.py, line 392
world_size = 8  # Change this to your GPU count (e.g., 2, 4, 8)
```

Then run:

```bash
# Activate the conda environment
conda activate dl-exp

# Run distributed training
python train_distributed.py
```

The distributed training uses PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training.

### Model Architecture

The model consists of:
1. **Embedding Layer**: Converts word indices to dense vectors
2. **BiLSTM Layers**: Multiple bidirectional LSTM layers with optional:
   - Residual connections
   - Layer normalization
   - Dropout
3. **Attention Mechanism**: Computes weighted attention over LSTM outputs
4. **Classification Layer**: Final linear layer for 5-class classification

### Output and Results

After training, you will find:

**In `saved_models/models/`**:
- `{experiment_name}_best_val_acc.pth`: Best model checkpoint

**In `saved_models/figures/`**:
- `{experiment_name}_loss.png`: Training and validation loss curves
- `{experiment_name}_acc.png`: Training and validation accuracy curves
- `comparison_val_acc.png`: Comparison of all experiments

### Experiment Results

The training script runs multiple experiments to compare different configurations:
- Baseline configuration
- Different dropout rates
- Different network depths
- With/without residual connections
- With/without layer normalization
- With/without learning rate decay

Final results will be printed showing the best configuration and accuracy.

### Troubleshooting

**CUDA Out of Memory**:
- Reduce `batch_size` in `config.yaml`
- Reduce `max_len` or `hidden_dim`
- Use fewer LSTM layers (`n_layers`)

**Dataset Not Found**:
- Verify files exist in `dataset/` directory
- Check file paths in `config.yaml`

**Multi-GPU Issues**:
- Ensure `world_size` matches your GPU count
- Check CUDA is available: `python -c "import torch; print(torch.cuda.device_count())"`

---

<a name="chinese"></a>
## 中文

### 项目概述

本项目实现了一个带注意力机制的双向LSTM（BiLSTM）模型，用于Yelp评论数据的情感分类。该模型使用深度学习技术将评论分类为5个星级（1-5星）。

### 特性

- **BiLSTM架构**：双向LSTM以更好地理解上下文
- **注意力机制**：加权注意力关注重要单词
- **灵活配置**：通过YAML文件配置多个超参数
- **分布式训练**：支持使用PyTorch DDP的多GPU训练
- **实验跟踪**：比较不同的模型配置
- **可视化**：自动生成训练曲线和对比图

### 前置要求

- Anaconda 或 Miniconda
- 支持CUDA的NVIDIA GPU（推荐）
- Python 3.11
- PyTorch with CUDA 12.1

### 目录结构设置

开始之前，创建必要的目录：

```bash
# 创建数据集目录
mkdir dataset

# 创建saved_models目录及其子目录
mkdir -p saved_models/figures
mkdir -p saved_models/models
```

设置完成后，您的目录结构应如下所示：
```
BiLstm-with-attention/
├── dataset/                          # 数据集目录
│   ├── yelp_academic_dataset_review.json  # 训练数据（需下载）
│   └── test.json                     # 测试数据（需下载）
├── saved_models/                     # 模型输出目录
│   ├── figures/                      # 训练图表和可视化
│   └── models/                       # 保存的模型检查点
├── model/                            # 模型架构
│   └── Bilstm_attention.py
├── utils/                            # 工具函数
│   └── data.py
├── config.yaml                       # 配置文件
├── environment.yml                   # Conda环境文件
├── train.py                          # 单GPU训练脚本
└── train_distributed.py              # 多GPU分布式训练脚本
```

### 数据集下载

从以下GitHub仓库下载Yelp学术数据集：

**仓库地址**: [Yelp Data Challenge 2013](https://github.com/rekiksab/Yelp-Data-Challenge-2013/tree/master/yelp_challenge/yelp_phoenix_academic_dataset)

**步骤**：
1. 访问仓库：https://github.com/rekiksab/Yelp-Data-Challenge-2013
2. 导航到 `yelp_challenge/yelp_phoenix_academic_dataset/`
3. 下载以下文件：
   - `yelp_academic_dataset_review.json`（训练数据）
   - 任何测试数据文件并重命名为 `test.json`
4. 将下载的文件放置在 `dataset/` 目录中

或者，您可以使用 `wget` 或 `git`：

```bash
# 克隆仓库
git clone https://github.com/rekiksab/Yelp-Data-Challenge-2013.git

# 将数据集文件复制到您的dataset目录
cp Yelp-Data-Challenge-2013/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json dataset/

# 创建或复制测试数据
cp Yelp-Data-Challenge-2013/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json dataset/test.json
```

### 环境配置

创建并配置conda环境：

```bash
# 从environment.yml创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate dl-exp

# 如需要，安装额外的依赖
pip install pyyaml
```

`environment.yml` 文件包含：
- Python 3.11
- 支持CUDA 12.1的PyTorch（pytorch-cuda=12.1）
- TorchVision和TorchAudio
- 用于数据处理的Pandas
- 用于可视化的Matplotlib

### 配置

编辑 `config.yaml` 以自定义训练参数：

```yaml
base:
  train_path: dataset/yelp_academic_dataset_review.json
  test_path: dataset/test.json
  max_vocab_size: 50000
  max_len: 256
  min_freq: 5
  batch_size: 64
  epochs: 15
  learning_rate: 0.001
  device: cuda
  seed: 42
  embed_dim: 256
  hidden_dim: 256
  n_layers: 2
  dropout: 0.3
  use_layer_norm: true
  use_residual: true
  lr_decay: true
  num_classes: 5

experiments:
  Baseline: {}
  dropout_0.5:
    dropout: 0.5
  # 根据需要添加更多实验配置
```

### 训练

#### 单卡训练

在单个GPU上训练：

```bash
# 激活conda环境
conda activate dl-exp

# 运行训练
python train.py
```

脚本将：
- 加载和预处理数据集
- 从训练数据构建词汇表
- 运行 `config.yaml` 中定义的所有实验
- 将最佳模型保存到 `saved_models/models/`
- 在 `saved_models/figures/` 中生成训练图表

#### 多卡分布式训练

在多个GPU上进行分布式训练：

**重要提示**：运行前，请修改 `train_distributed.py` 中的 `world_size` 参数以匹配您的GPU数量。

```python
# 在 train_distributed.py 的第392行
world_size = 8  # 将此更改为您的GPU数量（例如：2、4、8）
```

然后运行：

```bash
# 激活conda环境
conda activate dl-exp

# 运行分布式训练
python train_distributed.py
```

分布式训练使用PyTorch的DistributedDataParallel（DDP）实现高效的多GPU训练。

### 模型架构

模型包含：
1. **嵌入层**：将词索引转换为密集向量
2. **BiLSTM层**：多个双向LSTM层，可选配置：
   - 残差连接
   - 层归一化
   - Dropout
3. **注意力机制**：计算LSTM输出的加权注意力
4. **分类层**：用于5类分类的最终线性层

### 输出和结果

训练后，您将找到：

**在 `saved_models/models/` 中**：
- `{实验名称}_best_val_acc.pth`：最佳模型检查点

**在 `saved_models/figures/` 中**：
- `{实验名称}_loss.png`：训练和验证损失曲线
- `{实验名称}_acc.png`：训练和验证准确率曲线
- `comparison_val_acc.png`：所有实验的对比

### 实验结果

训练脚本运行多个实验以比较不同配置：
- 基线配置
- 不同的dropout率
- 不同的网络深度
- 有/无残差连接
- 有/无层归一化
- 有/无学习率衰减

最终结果将打印显示最佳配置和准确率。

### 故障排除

**CUDA内存不足**：
- 在 `config.yaml` 中减小 `batch_size`
- 减小 `max_len` 或 `hidden_dim`
- 使用更少的LSTM层（`n_layers`）

**找不到数据集**：
- 验证文件存在于 `dataset/` 目录中
- 检查 `config.yaml` 中的文件路径

**多GPU问题**：
- 确保 `world_size` 与您的GPU数量匹配
- 检查CUDA是否可用：`python -c "import torch; print(torch.cuda.device_count())"`

---

## License

This project is for academic and educational purposes.

## Citation

If you use this code in your research, please cite accordingly.

## Contact

For questions or issues, please open an issue on GitHub.
