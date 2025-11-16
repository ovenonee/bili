import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. 加载数据
# ========================
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')  # 请替换为你的 CSV 文件名
# 假设列名是: filename, play_count, like_count, label
print("原始数据形状:", df.shape)
print(df.head())

# 加载图像和文本特征
visual_features = np.load(r'D:\bilitest\features\visual_X.npy')  # 形状: (N, D1)
text_features = np.load(r'D:\bilitest\features\text_X.npy')      # 形状: (N, D2)
assert len(df) == len(visual_features) == len(text_features), "样本数不一致！"

play_counts = df['play_count'].values
like_counts = df['like_count'].values

# 创建三分类标签
q25 = np.percentile(play_counts, 25)
q75 = np.percentile(play_counts, 75)

def classify_hotness(pc):
    if pc < q25:
        return '低热度'
    elif pc < q75:
        return '中热度'
    else:
        return '高热度'

hotness_labels = np.array([classify_hotness(pc) for pc in play_counts])

# 分离不同热度类别的数据
low_mask = hotness_labels == '低热度'
high_mask = hotness_labels == '高热度'

# 选择样本数相等的子集进行对比
n_samples = min(np.sum(low_mask), np.sum(high_mask))
low_indices = np.random.choice(np.where(low_mask)[0], n_samples, replace=False)
high_indices = np.random.choice(np.where(high_mask)[0], n_samples, replace=False)

# 提取特征
low_visual = visual_features[low_indices]
high_visual = visual_features[high_indices]

low_text = text_features[low_indices]
high_text = text_features[high_indices]

low_play = play_counts[low_indices]
high_play = play_counts[high_indices]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================
# 1. 可视化差异最大的视觉特征分布
# ========================
print("🔍 绘制差异最大的视觉特征分布...")

# 计算视觉特征差异
visual_diff = high_visual.mean(axis=0) - low_visual.mean(axis=0)
visual_top_idx = np.argsort(np.abs(visual_diff))[::-1][:4]  # 前4个差异最大的特征

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feat_idx in enumerate(visual_top_idx):
    axes[i].hist(low_visual[:, feat_idx], bins=50, alpha=0.7, label='低热度', density=True, color='blue')
    axes[i].hist(high_visual[:, feat_idx], bins=50, alpha=0.7, label='高热度', density=True, color='red')
    axes[i].set_title(f'视觉特征 {feat_idx} 分布\n差异: {visual_diff[feat_idx]:.3f}')
    axes[i].set_xlabel('特征值')
    axes[i].set_ylabel('密度')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"差异最大的4个视觉特征: {visual_top_idx}")
print(f"对应的差异值: {visual_diff[visual_top_idx]}")

# ========================
# 2. 分析不同标签下高热度视频比例
# ========================
print("\n🔍 分析不同标签下高热度视频比例...")

# 计算每个标签的高热度比例
labels = df['label'].unique()
high_ratio_by_label = {}

for label in labels:
    label_mask = df['label'] == label
    label_hotness = hotness_labels[label_mask]
    high_count = np.sum(label_hotness == '高热度')
    total_count = len(label_hotness)
    high_ratio = high_count / total_count if total_count > 0 else 0
    high_ratio_by_label[label] = high_ratio

# 可视化标签与热度关系
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：各标签高热度比例
labels_list = list(high_ratio_by_label.keys())
ratios_list = list(high_ratio_by_label.values())

bars = ax1.bar(range(len(labels_list)), ratios_list, color='skyblue')
ax1.set_xlabel('标签')
ax1.set_ylabel('高热度比例')
ax1.set_title('不同标签下高热度视频比例')
ax1.set_xticks(range(len(labels_list)))
ax1.set_xticklabels(labels_list, rotation=45)
ax1.grid(True, alpha=0.3)

# 在柱子上显示数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom')

# 右图：各标签样本数量
label_counts = df['label'].value_counts()
ax2.bar(range(len(label_counts)), label_counts.values, color='lightcoral')
ax2.set_xlabel('标签')
ax2.set_ylabel('样本数量')
ax2.set_title('各标签样本数量分布')
ax2.set_xticks(range(len(label_counts)))
ax2.set_xticklabels(label_counts.index, rotation=45)
ax2.grid(True, alpha=0.3)

# 在柱子上显示数值
for i, (label, count) in enumerate(label_counts.items()):
    ax2.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("各标签高热度比例:")
for label, ratio in high_ratio_by_label.items():
    print(f"  {label}: {ratio:.2%} (样本数: {np.sum(df['label']==label)})")

# ========================
# 3. 热度与标签关系详细分析
# ========================
print("\n🔍 热度与标签关系详细分析...")

# 创建热度-标签交叉表
cross_table = pd.crosstab(df['label'], hotness_labels, normalize='index')
print("各标签下各类别比例:")
print(cross_table)

# 可视化热度-标签关系热力图
plt.figure(figsize=(10, 6))
sns.heatmap(cross_table.T, annot=True, fmt='.2%', cmap='YlOrRd', 
            cbar_kws={'label': '比例'})
plt.title('热度等级与标签关系热力图')
plt.xlabel('标签')
plt.ylabel('热度等级')
plt.tight_layout()
plt.show()

# 按高热度比例排序的标签
sorted_labels = sorted(high_ratio_by_label.items(), key=lambda x: x[1], reverse=True)
print(f"\n按高热度比例排序的标签:")
for label, ratio in sorted_labels:
    print(f"  {label}: {ratio:.2%}")

# ========================
# 4. PCA可视化（增强版）
# ========================
print("\n🔍 PCA可视化高低热度分离度...")

# 对视觉特征进行PCA
pca = PCA(n_components=2)
visual_features_combined = np.vstack([low_visual, high_visual])
pca_result = pca.fit_transform(visual_features_combined)

# 标签
pca_labels = ['低热度'] * len(low_visual) + ['高热度'] * len(high_visual)

plt.figure(figsize=(10, 8))
for label in np.unique(pca_labels):
    mask = np.array(pca_labels) == label
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
               label=label, alpha=0.6, s=20)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
plt.title('视觉特征PCA降维对比 (低热度 vs 高热度)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"PCA解释的方差比例: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# ========================
# 5. 综合分析报告
# ========================
print(f"\n=== 高热度封面特征综合分析 ===")
print(f"1. 视觉特征: 前4个差异最大特征为 {visual_top_idx}")
print(f"2. 标签分析: {len(labels)} 个不同标签，高热度比例范围 {min(ratios_list):.2%} - {max(ratios_list):.2%}")
print(f"3. PCA分离度: 解释了 {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.2%} 的方差")
print(f"4. 最高热度标签: {sorted_labels[0][0]} ({sorted_labels[0][1]:.2%} 高热度)")
print(f"5. 最低热度标签: {sorted_labels[-1][0]} ({sorted_labels[-1][1]:.2%} 高热度)")