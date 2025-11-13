import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

# 清洗并计算CTR
def convert_number(x):
    return float(x.replace('万', '')) * 10000 if '万' in x else float(x)

df['play_count_num'] = df['play_count'].apply(convert_number)
df['like_count_num'] = df['like_count'].apply(convert_number)
df['ctr'] = df['like_count_num'] / df['play_count_num']

# 绘制分布
sns.histplot(df['ctr'], bins=30, kde=True, color='skyblue')
plt.axvline(df['ctr'].mean(), color='red', linestyle='--', 
            label=f'均值: {df["ctr"].mean():.3f}')
plt.title('CTR分布：大部分视频集中在0-5%区间', fontsize=14)
plt.legend()
plt.savefig('ctr_distribution.png', dpi=300)
plt.show()