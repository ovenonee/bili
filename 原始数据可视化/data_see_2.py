import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# ==================== 初始化 ====================
FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'
MY_FONT = FontProperties(fname=FONT_PATH, size=12)
print(f"✅ 已加载字体: {FONT_PATH}")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ==================== 加载并处理数据 ====================
df = pd.read_csv(r'D:\bilitest\merged_data\merged_data.csv')

# ！计算CTR（关键）
df['CTR'] = df['like_count'] / (df['play_count'] + 1)
df['CTR'] = df['CTR'].clip(0, 1)  # 限制在[0,1]

print(f"📊 数据量: {len(df)}")
print(f"📈 CTR范围: [{df['CTR'].min():.4f}, {df['CTR'].max():.4f}]")
print(f"🏷️ 标签数量: {df['label'].nunique()}")

# ==================== 图1：CTR分布 ====================
plt.figure(figsize=(12, 6))
sns.histplot(df['CTR'], bins=50, kde=True, color='#4A90E2')
mean_val, median_val = df['CTR'].mean(), df['CTR'].median()
plt.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.4f}')
plt.axvline(median_val, color='orange', linestyle='--', label=f'中位数: {median_val:.4f}')
plt.title("图1 封面点击率（CTR）初始分布 [1]", fontproperties=MY_FONT)
plt.xlabel("CTR值", fontproperties=MY_FONT)
plt.ylabel("频数", fontproperties=MY_FONT)
plt.legend(prop=MY_FONT)
plt.tight_layout()
plt.savefig('fig1.png', dpi=300)
plt.show()

# ==================== 图2：饼图 ====================
plt.figure(figsize=(8, 8))
label_counts = df['label'].value_counts()
wedges, labels_texts, autotexts = plt.pie(
    label_counts.values, 
    labels=label_counts.index, 
    autopct='%1.1f%%', 
    colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
    startangle=90
)
for text in labels_texts + autotexts:
    text.set_fontproperties(MY_FONT)
plt.title("图2 标签类别分布 [2]", fontproperties=MY_FONT)
plt.tight_layout()
plt.savefig('fig2.png', dpi=300)
plt.show()

# ==================== 图3：CTR箱线图（中文标签） ====================
plt.figure(figsize=(14, 6))  # 增大宽度

# 动态获取实际标签（前10个高频标签，避免太拥挤）
top_labels = df['label'].value_counts().head(10).index.tolist()
data_to_plot = [df[df['label'] == label]['CTR'].values for label in top_labels]

# 绘制箱线图
box_plot = plt.boxplot(data_to_plot, 
                       labels=top_labels,  # 使用真实中文标签
                       patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))

# 设置颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166', '#118AB2', '#EF476F', 
          '#06D6A0', '#073B4C', '#FF9A76', '#9B5DE5'][:len(top_labels)]
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# ！修复x轴标签字体（关键）
for label in plt.gca().get_xticklabels():
    label.set_fontproperties(MY_FONT)

plt.title("图3 不同标签的CTR箱线图（前10类） [3]", fontproperties=MY_FONT)
plt.ylabel("CTR", fontproperties=MY_FONT)
plt.xlabel("标签类别", fontproperties=MY_FONT)
plt.xticks(rotation=45, ha='right')  # 旋转标签避免重叠
plt.tight_layout()
plt.savefig('fig3.png', dpi=300)
plt.show()

# ==================== 图4：播放量-点赞量 ====================
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['play_count'], df['like_count'], 
                     c=df['CTR'], cmap='viridis', alpha=0.6, s=20)
plt.xscale('log')
plt.yscale('log')
cbar = plt.colorbar(scatter, label='CTR')
cbar.set_label('CTR', fontproperties=MY_FONT)
plt.title("图4 播放量-点赞量关系（颜色表示CTR） [4]", fontproperties=MY_FONT)
plt.xlabel("播放量（log）", fontproperties=MY_FONT)
plt.ylabel("点赞量（log）", fontproperties=MY_FONT)
plt.tight_layout()
plt.savefig('fig4.png', dpi=300)
plt.show()