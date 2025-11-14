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

from PIL import Image
import os

from PIL import Image
import os

def plot_figure5():
    # ==================== 关键配置 ====================
    # 1. 设置covers文件夹的绝对路径
    #    根据你的实际情况修改，例如：
    #    COVERS_DIR = r"D:\bilitest\covers"  # 绝对路径
    #    COVERS_DIR = "covers"                # 相对路径
    COVERS_DIR = r"D:\bilitest\merged_data\covers"  # ！请确保这个路径正确
    
    # 2. 检查CSV中的文件名格式（是否需要加后缀）
    #    如果CSV中是 "1jpg"，实际文件是 "1.jpg"
    #    需要添加后缀：fname = row['filename'].replace('jpg', '.jpg')
    #    如果CSV中已是完整文件名，则不需修改
    # ==================================================
    
    if not os.path.exists(COVERS_DIR):
        print(f"❌ 文件夹不存在: {os.path.abspath(COVERS_DIR)}")
        print("请修改 COVERS_DIR 变量为正确路径")
        return
    
    # 检查文件后缀
    sample_fname = df['filename'].iloc[0]
    has_extension = '.' in sample_fname
    print(f"文件名格式: {sample_fname} (是否含后缀: {has_extension})")
    
    # 创建3x5网格
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    # 获取前3个高频标签
    top3_labels = df['label'].value_counts().head(3).index.tolist()
    print(f"选取标签: {top3_labels}")
    
    for idx, label in enumerate(top3_labels):
        samples = df[df['label'] == label].head(5)
        
        for j, (_, row) in enumerate(samples.iterrows()):
            ax_idx = idx * 5 + j
            
            # 构建正确的文件路径
            filename = row['filename']
            if not has_extension:  # 如果CSV中没有后缀
                filename = filename.replace('jpg', '.jpg')
            
            img_path = os.path.join(COVERS_DIR, filename)
            
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[ax_idx].imshow(img)
                else:
                    # 显示文件名方便调试
                    axes[ax_idx].text(0.5, 0.5, f'图片缺失\n{filename}', 
                                    fontproperties=MY_FONT, 
                                    ha='center', va='center',
                                    fontsize=8)
                
                # 设置标题
                axes[ax_idx].set_title(f"CTR: {row['CTR']:.4f}", 
                                      fontproperties=MY_FONT, 
                                      fontsize=9)
                axes[ax_idx].axis('off')
                
                # 设置左侧标签
                if j == 0:
                    axes[ax_idx].set_ylabel(label, fontproperties=MY_FONT, fontsize=12)
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                axes[ax_idx].text(0.5, 0.5, '加载失败', 
                                fontproperties=MY_FONT, 
                                ha='center', va='center')
                axes[ax_idx].axis('off')
    
    # 隐藏未使用的子图
    total_plots_needed = len(top3_labels) * 5
    for i in range(total_plots_needed, len(axes)):
        axes[i].set_visible(False)
    
    # 主标题
    fig.suptitle("图5 三标签封面样本对比 [5]", fontproperties=MY_FONT, fontsize=16)
    plt.tight_layout()
    plt.savefig('fig5_sample_grid.png', dpi=300)
    plt.show()

# 执行
plot_figure5()