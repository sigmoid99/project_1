# Predicting the price of Huawei

我们需要预测不同型号品牌的华为手机价格，使用多层感知机（MLP）模型。

## Installation

创建相应的虚拟环境以及下载需要的包

```python
conda create --name perdict_hw python=3.10
conda activate perdict_hw

conda update -n base -c defaults conda
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

数据集来自  ，按照80/20的划分方式，Excel表格应包含如下结构：

| x     | 1    | 2    | 3    | 4    | 5    | 6    |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- |
| $y_1$ | 4    | 3    | 2    | 12   | 3    |      |
| $y_2$ | 1    | 22   | 3    |      |      |      |

- 第一行包含特征 $x$。
- 后续行包含目标 $y_i$，每一行代表不同的目标函数.

## Structure

- `model.py`: 定义多层感知机（MLP）模型的结构。

- `train.py`: 包含模型训练的相关函数。

- `predict.py`: 定义预测函数。

- `evaluate.py`: 主脚本，用于读取数据、训练模型、进行预测和输出结果。

## Example

以下是 `evaluate.py` 的核心代码：

```python
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import MLP
from train import train_model
from predict import predict

if __name__ == "__main__":
    data = pd.read_excel('data.xlsx', header=0)
    x_data = data.iloc[0, 1:].values  # 特征 x
    y_index = 1  # 指定 y_i 行索引
    y_target = data.iloc[y_index, 1:].values  # 获取指定 y_i

    # 数据处理和模型训练...
```

### Output

运行脚本后，程序将输出指定目标$y_i$ 与特征 $x$ 的预测结果，如下所示：

```css
Predicted y_2 for input [1, 2, 3, 4, 5, 6] = 10.3456
```

## Evaluation

脚本中包含 AUC ROC 的计算，以评估模型的性能。根据需要，可以调整计算方式。

## conclusion

该示例文档提供了完整的使用指南，帮助用户理解如何使用该项目进行目标函数的预测。如果需要更多功能或遇到问题，请联系开发者。

------

你可以根据自己的需要进一步调整和扩展这个文档。如果有其他要求，欢迎随时告诉我！

