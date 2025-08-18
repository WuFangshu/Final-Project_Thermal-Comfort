import pandas as pd
import numpy as np
import os

# === 路径配置 ===
csv_path = r"E:\thermal_project\xlstm\autotherm_data\train.csv"
save_dir = r"E:\thermal_project\xlstm\data"
os.makedirs(save_dir, exist_ok=True)

# === 加载 CSV 数据 ===
df = pd.read_csv(csv_path, low_memory=False)

# === 丢弃非数值列（保留所有数值特征）===
df_numeric = df.select_dtypes(include=[np.number])

# === 确保标签在列中 ===
assert 'Label' in df_numeric.columns, "Label 列未被识别为数值型，请检查数据格式"

# === 所有特征列（除去 Label）===
feature_cols = [col for col in df_numeric.columns if col != 'Label']

# === 滑动窗口生成样本（处理前10万个）===
sequence_length = 30
max_samples = 100000  # 你可以调大一点，比如 200000、300000
x_seq = []
y_label = []

limit = min(len(df_numeric) - sequence_length, max_samples)

for i in range(limit):
    x_window = df_numeric.iloc[i:i+sequence_length][feature_cols].values
    y = df_numeric.iloc[i+sequence_length-1]['Label']
    x_seq.append(x_window)
    y_label.append(y)

x_seq = np.array(x_seq, dtype=np.float32)  # 改为 float32 节省一半内存
y_label = np.array(y_label)

# === 保存为 .npy 文件 ===
np.save(os.path.join(save_dir, "x_seq_small.npy"), x_seq)
np.save(os.path.join(save_dir, "y_label_small.npy"), y_label)

print("✅ 数据预处理完成！（部分样本）")
print("x_seq shape:", x_seq.shape)
print("y_label shape:", y_label.shape)
