import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import MLP
from train import train_model
from predict import predict

if __name__ == "__main__":
    # 读取数据
    data = pd.read_excel('..\datasets\data.xlsx', header=0)  # 替换为你的Excel文件名
    x_data = data.iloc[0, 1:].values  # 第一行作为特征 x

    # 指定要分析的 y_i 行索引
    y_index = 1  # 例如，指定 y_1 为索引 1
    y_target = data.iloc[y_index, 1:].values  # 获取指定的 y_i

    # 将 y_target 转换为 Tensor
    y_target_tensor = torch.FloatTensor(y_target).view(-1, 1)  # 列向量

    # 创建数据加载器
    x_tensor = torch.FloatTensor(x_data).unsqueeze(0)  # 将 x 转换为一行的 Tensor
    dataset = TensorDataset(x_tensor.repeat(y_target.shape[0], 1), y_target_tensor)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型、损失函数和优化器
    input_size = x_tensor.shape[1]
    model = MLP(input_size)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    # 进行预测
    predictions = predict(model, x_tensor.repeat(y_target.shape[0], 1))

    print(f'Relationship between x and y_{y_index + 1}:')
    for i in range(len(predictions)):
        print(f'Predicted y_{y_index + 1} for input {x_data} = {predictions[i].item():.4f}')

    # 评估 AUC ROC（可选）
    # 在此示例中，可能需要根据需求调整 AUC ROC 的计算方式
