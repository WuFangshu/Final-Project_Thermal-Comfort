import pandas as pd

# read parquet document
df = pd.read_parquet('test1.parquet')

# save as csv
df.to_csv('test1.csv', index=False)