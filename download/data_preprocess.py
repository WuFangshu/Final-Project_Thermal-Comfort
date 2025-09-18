import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet('test1.parquet')

# 保存为 csv
df.to_csv('E:/Final_Project_Thermal_Comfort/autotherm_data/Raw/train-00000-of-00005.csv', index=False)