import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import pandas as pd

# 修改为你的 CSV 路径
csv_path = r"autotherm_data/train.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path, low_memory=False)

# 打印前几行数据，检查数据结构
print("== 前几行 ==")
print(df.head(10).to_string())

# 检查 Label 列基本情况
print("\n== 标签统计 ==")
if "Label" in df.columns:
    print("标签总数:", df["Label"].nunique())
    print("每个标签出现次数:")
    print(df["Label"].value_counts().sort_index())
    print("\n前30个标签值:")
    print(df["Label"].head(30).tolist())
else:
    print("❌ CSV 中没有找到名为 'Label' 的列！")
