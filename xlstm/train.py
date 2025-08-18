import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
import numpy as np

# 加载 .npy 文件 
x = np.load("data/x_seq_small.npy")  # 形状应为 (N, T, D)
y = np.load("data/y_label_small.npy")  # 形状应为 (N,)

# 可选：标签平移 [-3,3] -> [0,6]
y = y.astype(np.int64) + 3

# 自动填充最后维度为 4 的倍数
seq_len, feature_dim = x.shape[1], x.shape[2]
if feature_dim % 4 != 0:
    new_dim = ((feature_dim + 3) // 4) * 4
    print(f"embedding_dim={feature_dim} 不是4的倍数，自动填充为 {new_dim}")
    padded = np.zeros((x.shape[0], x.shape[1], new_dim), dtype=np.float32)
    padded[:, :, :feature_dim] = x
    x = padded

# 构建 Dataset
X = torch.tensor(x)
Y = torch.tensor(y)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 构建模型配置 
mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()

model_config = xLSTMBlockStackConfig(
    embedding_dim=X.shape[2],
    num_blocks=2,
    slstm_at=[1],
    dropout=0.1,
    bias=False,
    mlstm_block=mlstm_config,
    slstm_block=slstm_config,
    context_length=X.shape[1]
)

# 初始化模型 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMBlockStack(model_config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练 
model.train()
for epoch in range(10):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        out = model(batch_x)

        # 如果是序列输出 (B, T, C)，只取最后一步
        if out.dim() == 3:
            out = out[:, -1, :]

        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
