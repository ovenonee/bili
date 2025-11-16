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

# 分离高低热度数据
low_mask = hotness_labels == '低热度'
high_mask = hotness_labels == '高热度'

low_visual = visual_features[low_mask]
high_visual = visual_features[high_mask]

# 计算差异最大的特征
visual_diff = high_visual.mean(axis=0) - low_visual.mean(axis=0)
top_feature_indices = np.argsort(np.abs(visual_diff))[::-1][:10]

print(f"差异最大的10个视觉特征: {top_feature_indices}")
print(f"对应的差异值: {visual_diff[top_feature_indices]}")

# ========================
# 2. 尝试还原特征含义（基于常见视觉特征提取方法）
# ========================
def interpret_visual_features(feature_idx, feature_diff_value):
    """
    尝试解释视觉特征的含义
    这里提供一些常见的视觉特征解释模板
    """
    interpretations = {
        # 颜色相关特征（常见于颜色直方图、颜色矩等）
        'color': [
            '红色通道强度', '绿色通道强度', '蓝色通道强度',
            '色调分布', '饱和度分布', '亮度分布',
            '颜色多样性', '主色调', '对比度'
        ],
        # 纹理相关特征（常见于LBP、GLCM等）
        'texture': [
            '纹理复杂度', '局部纹理模式', '灰度共生矩阵特征',
            '边缘密度', '平滑度', '粗糙度'
        ],
        # 形状/构图相关特征（常见于HOG、SIFT等）
        'shape': [
            '水平边缘', '垂直边缘', '对角线边缘',
            '圆形特征', '直线特征', '构图复杂度',
            '主体位置', '前景背景对比'
        ],
        # 深度学习特征（常见于CNN）
        'cnn': [
            '人脸检测响应', '文字检测响应', '物体识别特征',
            '场景分类特征', '风格识别特征'
        ]
    }
    
    # 基于特征索引范围判断特征类型
    if feature_idx < 50:  # 假设前50是颜色特征
        feature_type = 'color'
        feature_name = interpretations['color'][feature_idx % len(interpretations['color'])]
    elif feature_idx < 100:  # 假设50-100是纹理特征
        feature_type = 'texture'
        feature_name = interpretations['texture'][feature_idx % len(interpretations['texture'])]
    elif feature_idx < 200:  # 假设100-200是形状特征
        feature_type = 'shape'
        feature_name = interpretations['shape'][feature_idx % len(interpretations['shape'])]
    else:  # 假设200+是CNN特征
        feature_type = 'cnn'
        feature_name = interpretations['cnn'][feature_idx % len(interpretations['cnn'])]
    
    return feature_type, feature_name

# ========================
# 3. 可视化特征差异的统计信息
# ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建特征解释和可视化
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.ravel()

for i, feat_idx in enumerate(top_feature_indices[:6]):  # 显示前6个
    feature_type, feature_name = interpret_visual_features(feat_idx, visual_diff[feat_idx])
    
    # 绘制该特征在高低热度间的分布对比
    ax = axes[i]
    ax.hist(low_visual[:, feat_idx], bins=50, alpha=0.7, label='低热度', density=True, color='blue')
    ax.hist(high_visual[:, feat_idx], bins=50, alpha=0.7, label='高热度', density=True, color='red')
    
    ax.set_title(f'特征 {feat_idx}: {feature_name}\n类型: {feature_type}\n差异: {visual_diff[feat_idx]:.3f}')
    ax.set_xlabel('特征值')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================
# 4. 特征类型分析
# ========================
feature_types = []
for feat_idx in top_feature_indices:
    feature_type, feature_name = interpret_visual_features(feat_idx, visual_diff[feat_idx])
    feature_types.append(feature_type)

print(f"\n=== 特征类型分布 ===")
for feat_type in ['color', 'texture', 'shape', 'cnn']:
    count = feature_types.count(feat_type)
    if count > 0:
        print(f"{feat_type}类特征: {count}个")

# ========================
# 5. 创建特征重要性雷达图
# ========================
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib import pyplot as plt
import numpy as np

def create_radar_chart(feature_indices, feature_diffs, feature_names):
    # 选择前8个最重要的特征
    top_8_indices = feature_indices[:8]
    top_8_diffs = feature_diffs[top_8_indices]
    top_8_names = [interpret_visual_features(idx, diff)[1] for idx, diff in zip(top_8_indices, top_8_diffs)]
    
    # 归一化差异值用于可视化
    angles = np.linspace(0, 2 * np.pi, len(top_8_names), endpoint=False).tolist()
    values = np.abs(top_8_diffs).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='red')
    ax.fill(angles, values, alpha=0.25, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_8_names, fontsize=10)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title('高热度封面关键视觉特征雷达图', size=16, pad=20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return top_8_names, top_8_diffs

top_features, top_diffs = create_radar_chart(top_feature_indices, visual_diff, 
                                            [interpret_visual_features(idx, diff)[1] 
                                             for idx, diff in zip(top_feature_indices, visual_diff)])

print(f"\n=== 重要特征分析 ===")
for name, diff in zip(top_features, top_diffs):
    direction = "高热度更高" if diff > 0 else "低热度更高"
    print(f"{name}: 差异={diff:.3f} ({direction})")

# ========================
# 6. 特征组合分析
# ========================
print(f"\n=== 高热度封面视觉特征模式 ===")

# 分析哪些特征组合对高热度重要
positive_features = []  # 高热度更高的特征
negative_features = []  # 低热度更高的特征

for idx, diff in zip(top_feature_indices, visual_diff[top_feature_indices]):
    feature_type, feature_name = interpret_visual_features(idx, diff)
    if diff > 0:
        positive_features.append((feature_name, diff))
    else:
        negative_features.append((feature_name, diff))

print(f"\n高热度封面特征 (值更高):")
for name, diff in positive_features[:5]:
    print(f"  • {name}: +{diff:.3f}")

print(f"\n低热度封面特征 (值更高):")
for name, diff in negative_features[:5]:
    print(f"  • {name}: {diff:.3f}")

# ========================
# 7. 实际应用建议
# ========================
print(f"\n=== 封面优化建议 ===")
print(f"高热度封面可能具备的视觉特征:")
print(f"• 特定的颜色组合/色调 (特征 {top_feature_indices[0]})")
print(f"• 特定的纹理模式 (特征 {top_feature_indices[1]})")
print(f"• 特定的构图方式 (特征 {top_feature_indices[2]})")
print(f"• 特定的主体检测模式 (特征 {top_feature_indices[3]})")
print(f"\n建议UP主在设计封面时关注这些视觉元素的差异！")

# ========================
# 8. 特征重要性条形图
# ========================
plt.figure(figsize=(12, 8))
top_10_names = [interpret_visual_features(idx, diff)[1] for idx, diff in zip(top_feature_indices[:10], visual_diff[top_feature_indices[:10]])]
top_10_diffs = visual_diff[top_feature_indices[:10]]

colors = ['red' if x > 0 else 'blue' for x in top_10_diffs]
bars = plt.barh(range(len(top_10_names)), top_10_diffs, color=colors, alpha=0.7)
plt.yticks(range(len(top_10_names)), top_10_names)
plt.xlabel('特征差异 (高热度 - 低热度)')
plt.title('高热度封面关键视觉特征差异 (前10个)')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# 在条形图上显示数值
for i, (bar, diff) in enumerate(zip(bars, top_10_diffs)):
    plt.text(diff, i, f'{diff:.3f}', ha='left' if diff > 0 else 'right', va='center')

plt.tight_layout()
plt.show()