import pandas as pd

df = pd.read_parquet('parquet')

df.to_csv('E:/Final_Project_Thermal_Comfort/autotherm_data/Raw/train-00000-of-00005.csv', index=False)
