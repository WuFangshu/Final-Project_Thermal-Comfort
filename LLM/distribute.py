import pandas as pd
import matplotlib.pyplot as plt

# df1 = pd.read_csv("E:/thermal_project/compare/gan.csv")
# df2 = pd.read_csv("E:/thermal_project/compare/noise.csv")
df1 = pd.read_csv("E:/123/llm/llm_out.csv")
df2 = pd.read_csv("E:/123/llm/train.csv")

# Choose the column you want to visualize
column = "Bodytemp"

plt.figure(figsize=(8, 6))

# Plot histogram for synthetic
plt.hist(df1[column].dropna(), bins=10, density=True, alpha=0.25, edgecolor="black", label="LLM Synthetic")

# Plot histogram for real
plt.hist(df2[column].dropna(), bins=10, density=True, alpha=0.25, edgecolor="black", label="Real Data")


plt.xlabel(column)
plt.ylabel("Density")
plt.title(f"Histogram of {column} (Synthetic vs Real)")
plt.legend()
plt.tight_layout()
plt.show()




