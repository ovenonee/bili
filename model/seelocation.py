import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# 1. 读数据 & 特征
# --------------------------------------------------
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')
visual_features = np.load(r'D:\bilitest\features\visual_X.npy')
text_features   = np.load(r'D:\bilitest\features\text_X.npy')

assert len(df) == len(visual_features) == len(text_features), '样本数不一致！'

play_counts = df['play_count'].values
print('原始数据形状:', df.shape)
print('视觉特征维度 :', visual_features.shape[1])
print('文本特征维度 :', text_features.shape[1])

# --------------------------------------------------
# 2. 三分类标签 & 高低热度 mask
# --------------------------------------------------
q25, q75 = np.percentile(play_counts, [25, 75])

def classify_hotness(pc):
    if pc < q25:
        return '低热度'
    elif pc < q75:
        return '中热度'
    return '高热度'

hotness_labels = np.array([classify_hotness(pc) for pc in play_counts])
low_mask  = hotness_labels == '低热度'
high_mask = hotness_labels == '高热度'
low_visual  = visual_features[low_mask]
high_visual = visual_features[high_mask]

# --------------------------------------------------
# 3. 构图特征提取
# --------------------------------------------------
def analyze_composition_features():
    if visual_features.shape[1] < 40:
        print('视觉特征维度较小，可能不包含构图特征')
        return None, None, None, None
    start, end = 40, visual_features.shape[1]
    comp_idx = list(range(start, end))
    print(f'分析构图特征范围: {start}-{end-1} (共{len(comp_idx)}个)')
    low_comp  = low_visual[:, comp_idx]
    high_comp = high_visual[:, comp_idx]
    diffs = high_comp.mean(axis=0) - low_comp.mean(axis=0)
    return comp_idx, diffs, low_comp, high_comp

comp_indices, comp_diffs, low_comp, high_comp = analyze_composition_features()

# --------------------------------------------------
# 4. 画图
# --------------------------------------------------
if comp_indices is not None and len(comp_indices) > 0:

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ---------- 4.1 差异条形图 ----------
    fig1, ax_bar = plt.subplots(1, 1, figsize=(8, 6))
    top15 = np.argsort(np.abs(comp_diffs))[::-1][:15]
    colors = ['red' if comp_diffs[i] > 0 else 'blue' for i in top15]
    ax_bar.barh(range(len(top15)), comp_diffs[top15], color=colors, alpha=0.7)
    ax_bar.set_yticks(range(len(top15)))
    ax_bar.set_yticklabels([f'特征_{comp_indices[i]}' for i in top15])
    ax_bar.set_xlabel('差异 (高热度 - 低热度)')
    ax_bar.set_title('构图特征差异 (前15个)')
    ax_bar.axvline(0, color='k', ls='--', alpha=0.5)
    ax_bar.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 4.2 分布对比（3 张独立图） ----------
    top3 = np.argsort(np.abs(comp_diffs))[::-1][:3]
    for k, local_idx in enumerate(top3):
        global_idx = comp_indices[local_idx]
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.hist(low_comp[:, local_idx], bins=50, density=True, alpha=0.7, label='低热度', color='blue')
        ax.hist(high_comp[:, local_idx], bins=50, density=True, alpha=0.7, label='高热度', color='red')
        ax.set_title(f'构图特征 {global_idx} 分布')
        ax.set_xlabel('特征值')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ---------- 4.3 复杂度 ----------
    low_complex  = low_comp.var(axis=1)
    high_complex = high_comp.var(axis=1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    ax2[0].hist(low_complex, bins=50, density=True, alpha=0.7, label='低热度', color='blue')
    ax2[0].hist(high_complex, bins=50, density=True, alpha=0.7, label='高热度', color='red')
    ax2[0].set_xlabel('构图复杂度 (方差)')
    ax2[0].set_title('构图复杂度分布对比')
    ax2[0].legend()
    ax2[0].grid(alpha=0.3)

    ax2[1].scatter(low_complex, high_complex, alpha=0.5)
    lim = [min(low_complex.min(), high_complex.min()), max(low_complex.max(), high_complex.max())]
    ax2[1].plot(lim, lim, 'k--', alpha=0.5)
    ax2[1].set_xlabel('低热度构图复杂度')
    ax2[1].set_ylabel('高热度构图复杂度')
    ax2[1].set_title('构图复杂度相关性')
    ax2[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print('\n=== 构图复杂度分析 ===')
    print(f'低热度: {low_complex.mean():.4f} ± {low_complex.std():.4f}')
    print(f'高热度: {high_complex.mean():.4f} ± {high_complex.std():.4f}')
    print(f'高热度 / 低热度 = {high_complex.mean()/low_complex.mean():.2f} 倍')

    # ---------- 4.4 重要性排名 ----------
    sort_idx = np.argsort(np.abs(comp_diffs))[::-1]
    print('\n=== 构图特征重要性（前10）===')
    for k in range(min(10, len(sort_idx))):
        idx = sort_idx[k]
        print(f'{k+1:2d}. 特征_{comp_indices[idx]}: '
              f'差异={comp_diffs[idx]:.4f} （{"高热度更高" if comp_diffs[idx]>0 else "低热度更高"}）')
    pos = (comp_diffs > 0).sum()
    neg = (comp_diffs < 0).sum()
    print(f'\n高热度偏好特征数: {pos}   低热度偏好特征数: {neg}')

    # ---------- 4.5 热力图 ----------
    top20 = sort_idx[:min(20, len(sort_idx))]
    cmp_mat = np.vstack([low_comp[:, top20].mean(axis=0),
                         high_comp[:, top20].mean(axis=0)])
    plt.figure(figsize=(14, 4))
    sns.heatmap(cmp_mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=[f'特征_{comp_indices[i]}' for i in top20],
                yticklabels=['低热度', '高热度'])
    plt.xticks(rotation=45)
    plt.title('高低热度构图特征对比热力图')
    plt.tight_layout()
    plt.show()

    print('\n=== 构图分析总结 ===')
    avg_abs = np.abs(comp_diffs).mean()
    sig_num = (np.abs(comp_diffs) > avg_abs).sum()
    print(f'1. 平均绝对差异: {avg_abs:.4f}')
    print(f'2. 高/低热度复杂度倍率: {high_complex.mean()/low_complex.mean():.2f}')
    print(f'3. 显著差异特征数: {sig_num}')
    print('4. 高热度封面在空间/布局特征上呈现不同模式')

# --------------------------------------------------
# 5. 若无构图特征，整体差异
# --------------------------------------------------
else:
    print('\n=== 整体视觉特征分析 ===')
    all_diff = high_visual.mean(axis=0) - low_visual.mean(axis=0)
    top20 = np.argsort(np.abs(all_diff))[::-1][:20]
    for k, idx in enumerate(top20):
        print(f'{k+1:2d}. 特征_{idx}: 差异={all_diff[idx]:.4f} '
              f'（{"高热度更高" if all_diff[idx]>0 else "低热度更高"}）')