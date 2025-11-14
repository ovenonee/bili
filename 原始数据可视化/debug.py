import pandas as pd

df = pd.read_csv(r'D:\bilitest\merged_data\merged_data.csv')
print("=== 数据诊断 ===")
print(f"标签列名: {df.columns}")
print(f"标签唯一值: {df['label'].unique()}")
print(f"标签计数:\n{df['label'].value_counts()}")
print(f"CTR统计: 均值={df['CTR'].mean()}, 范围=[{df['CTR'].min()}, {df['CTR'].max()}]")