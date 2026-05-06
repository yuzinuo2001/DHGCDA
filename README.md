
## 文件内容

- `main_GPU.py`
  5 折训练与测试主程序，负责数据读取、模型训练、验证、测试和结果统计。
- `model_GPU.py`
  模型定义文件，包含生物特征编码、双通道超图编码、多视图融合与配对解码模块。
- `build_hyg.py`
  多视图超图构建脚本，用于生成训练所需的超图 `pickle` 文件。
- `data/`
  整理版脚本当前依赖的数据目录。

## 数据目录要求

默认依赖 `data/` 目录下的以下内容：

- circRNA 与 disease ID 文件
- embedding 文件
- `CD_data/` 中的训练/验证/测试划分文件
- `hyg_new/` 中的超图 `pickle` 文件
- circRNA 相似性文件
- miRNA-disease 关联文件

## 使用方法

### 1. 构建超图

如果需要重新生成超图输入文件，可运行：

```bash
python build_hyg.py
```
运行后会在 `hyg_new/` 下生成 `hyg_dict0.pickle` 到 `hyg_dict4.pickle`。

### 2. 训练与测试

运行主程序：

```bash
python main_GPU.py
```

`main_GPU.py` 支持的主要可选参数包括：

- `--ablation`
- `--ablation_mode`
- `--dual_ablation`
- `--view_fusion`
- `--view_drop`
- `--use_view_gate`
- `--hgnn_dim`
- `--lr`
- `--exp_tag`

当前脚本中的默认核心设置为：

- `epochs = 100`
- `lr = 0.004`
- `bio_out_dim = 32`
- `hgnn_dim = 512`
- 设备优先使用 CUDA，没有 GPU 时回退到 CPU

## 依赖环境

脚本至少依赖以下 Python 包：

- `torch`
- `torch_geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## 说明与注意事项

- `main_GPU.py` 通过 `from model_GPU import *` 导入同目录下的 `model_GPU.py`，因此两者应保持在同一目录中
- `build_hyg.py` 中仍保留了部分原始环境下的绝对路径设置。如果要在其他机器上运行，建议将这些路径手动改为可移植的相对路径
