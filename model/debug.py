import numpy as np

path = r'D:\bilitest\features\visual_X.npy'
n_row, n_col = np.load(path, mmap_mode='r').shape
print('行数 =', n_row, '  列数 =', n_col)


import pandas as pd
import numpy as np

CLEANED_CSV = r'D:\bilitest\cleaned_data\result.csv'   # 你的路径

df = pd.read_csv(CLEANED_CSV)
print('实际 csv 行数（含表头）:', len(df))
print('表头:', df.columns.tolist())

# 如果是表头导致 13401，直接跳过表头再读一次
df = pd.read_csv(CLEANED_CSV, skiprows=1)   # 跳过第一行表头
print('跳过表头后行数:', len(df))

# 如果你不想跳过表头，就改成 13401 即可
assert len(df) == 13400, f'csv 行数={len(df)}，不等于 13400！'