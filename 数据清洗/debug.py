import pandas as pd
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')

# 去掉 play_count 或 like_count 为 0 的记录
df = df[(df['play_count'] > 0) & (df['like_count'] > 0)]

for col in ['play_count', 'like_count']:
    zero_mask = df[col] == 0
    rate = zero_mask.sum() / len(df)
    print(f'{col} 0 值占比 {rate:.2%}')

    if zero_mask.any():
        print(df.loc[zero_mask, 'filename'].sample(min(5, zero_mask.sum())).tolist())
    else:
        print(f'清洗后已无 {col} == 0 的记录！')
    print('------')

""" import pandas as pd

csv_path = r'D:\bilitest\cleaned_data\result.csv'
print('数据行数（不含表头）:', len(pd.read_csv(csv_path))) """