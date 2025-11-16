import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# 创建三分类标签（使用原始分位数）
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
mid_mask = hotness_labels == '中热度'
high_mask = hotness_labels == '高热度'

print(f"热度分布: 低({np.sum(low_mask)}), 中({np.sum(mid_mask)}), 高({np.sum(high_mask)})")

# ========================
# 2. 高热度 vs 低热度的特征对比
# ========================
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

low_like = like_counts[low_indices]
high_like = like_counts[high_indices]

print(f"\n对比样本数: 低热度 {len(low_indices)}, 高热度 {len(high_indices)}")

# ========================
# 3. 统计特征分析
# ========================
print("\n=== 播放量统计 ===")
print(f"低热度播放量: {low_play.mean():.0f} ± {low_play.std():.0f}")
print(f"高热度播放量: {high_play.mean():.0f} ± {high_play.std():.0f}")
print(f"高热度是低热度的 {high_play.mean()/low_play.mean():.1f} 倍")

print(f"\n=== 点赞量统计 ===")
print(f"低热度点赞量: {low_like.mean():.0f} ± {low_like.std():.0f}")
print(f"高热度点赞量: {high_like.mean():.0f} ± {high_like.std():.0f}")
print(f"高热度是低热度的 {high_like.mean()/low_like.mean():.1f} 倍")

print(f"\n=== 点赞/播放比率 ===")
low_ratio = low_like / (low_play + 1)
high_ratio = high_like / (high_play + 1)
print(f"低热度点赞/播放比: {low_ratio.mean():.3f} ± {low_ratio.std():.3f}")
print(f"高热度点赞/播放比: {high_ratio.mean():.3f} ± {high_ratio.std():.3f}")

# ========================
# 4. 可视化分析
# ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 播放量和点赞量分布对比
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 播放量分布
axes[0,0].hist(low_play, bins=50, alpha=0.7, label='低热度', density=True)
axes[0,0].hist(high_play, bins=50, alpha=0.7, label='高热度', density=True)
axes[0,0].set_xlabel('播放量')
axes[0,0].set_ylabel('密度')
axes[0,0].set_title('播放量分布对比')
axes[0,0].legend()

# 点赞量分布
axes[0,1].hist(low_like, bins=50, alpha=0.7, label='低热度', density=True)
axes[0,1].hist(high_like, bins=50, alpha=0.7, label='高热度', density=True)
axes[0,1].set_xlabel('点赞量')
axes[0,1].set_ylabel('密度')
axes[0,1].set_title('点赞量分布对比')
axes[0,1].legend()

# 点赞/播放比率分布
axes[1,0].hist(low_ratio, bins=50, alpha=0.7, label='低热度', density=True)
axes[1,0].hist(high_ratio, bins=50, alpha=0.7, label='高热度', density=True)
axes[1,0].set_xlabel('点赞/播放比率')
axes[1,0].set_ylabel('密度')
axes[1,0].set_title('点赞/播放比率分布对比')
axes[1,0].legend()

# 播放量 vs 点赞量散点图
axes[1,1].scatter(low_play, low_like, alpha=0.5, label='低热度', s=10)
axes[1,1].scatter(high_play, high_like, alpha=0.5, label='高热度', s=10)
axes[1,1].set_xlabel('播放量')
axes[1,1].set_ylabel('点赞量')
axes[1,1].set_title('播放量 vs 点赞量')
axes[1,1].legend()
axes[1,1].set_yscale('log')
axes[1,1].set_xscale('log')

plt.tight_layout()
plt.show()

# ========================
# 5. 视觉特征PCA分析
# ========================
# 对视觉特征进行PCA降维以便可视化
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

print(f"\nPCA解释的方差比例: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# ========================
# 6. 特征重要性分析（通过统计差异）
# ========================
# 计算视觉特征的均值差异
visual_diff = high_visual.mean(axis=0) - low_visual.mean(axis=0)
text_diff = high_text.mean(axis=0) - low_text.mean(axis=0)

# 找出差异最大的特征
n_top_features = 10

# 视觉特征差异
visual_top_idx = np.argsort(np.abs(visual_diff))[::-1][:n_top_features]
print(f"\n=== 视觉特征差异最大的前{n_top_features}个 ===")
for i, idx in enumerate(visual_top_idx):
    print(f"特征 {idx}: 高热度均值={high_visual[:, idx].mean():.3f}, 低热度均值={low_visual[:, idx].mean():.3f}, 差异={visual_diff[idx]:.3f}")

# 文本特征差异
text_top_idx = np.argsort(np.abs(text_diff))[::-1][:n_top_features]
print(f"\n=== 文本特征差异最大的前{n_top_features}个 ===")
for i, idx in enumerate(text_top_idx):
    print(f"特征 {idx}: 高热度均值={high_text[:, idx].mean():.3f}, 低热度均值={low_text[:, idx].mean():.3f}, 差异={text_diff[idx]:.3f}")

# ========================
# 7. 相关性分析
# ========================
# 计算播放量与各特征的相关性
correlations = []
feature_names = []

# 视觉特征相关性
for i in range(min(10, visual_features.shape[1])):  # 取前10个视觉特征
    corr = np.corrcoef(visual_features[:, i], play_counts)[0, 1]
    correlations.append(corr)
    feature_names.append(f'视觉_{i}')

# 文本特征相关性
for i in range(min(10, text_features.shape[1])):  # 取前10个文本特征
    corr = np.corrcoef(text_features[:, i], play_counts)[0, 1]
    correlations.append(corr)
    feature_names.append(f'文本_{i}')

# 可视化相关性
plt.figure(figsize=(12, 8))
top_corr_idx = np.argsort(np.abs(correlations))[::-1][:15]
top_correlations = [correlations[i] for i in top_corr_idx]
top_names = [feature_names[i] for i in top_corr_idx]

plt.barh(range(len(top_correlations)), top_correlations)
plt.yticks(range(len(top_names)), top_names)
plt.xlabel('与播放量的相关系数')
plt.title('特征与播放量的相关性 (前15个)')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================
# 8. 总结报告
# ========================
print(f"\n=== 高热度封面特征分析总结 ===")
print(f"1. 播放量: 高热度视频平均播放量是低热度的 {high_play.mean()/low_play.mean():.1f} 倍")
print(f"2. 点赞量: 高热度视频平均点赞量是低热度的 {high_like.mean()/low_like.mean():.1f} 倍")
print(f"3. 点赞率: 高热度视频点赞/播放比率 {high_ratio.mean():.3f} vs 低热度 {low_ratio.mean():.3f}")
print(f"4. 视觉差异: 找出了 {len(visual_top_idx)} 个视觉特征在高低热度间存在显著差异")
print(f"5. 文本差异: 找出了 {len(text_top_idx)} 个文本特征在高低热度间存在显著差异")
print(f"6. PCA分离度: 视觉特征在高低热度间存在一定程度的分离")