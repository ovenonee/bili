import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# 1. 读取清洗后的数据
try:
    csv_path = sys.argv[1]
except IndexError:
    csv_path = r'D:\bilitest\cleaned_data\result.csv'   # 默认路径，可改

df = pd.read_csv(csv_path)
out_dir = os.path.join(os.path.dirname(csv_path), 'report')
os.makedirs(out_dir, exist_ok=True)

# 2. 中文与风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# 3. 图1：播放量分布（对数坐标）
plt.figure(figsize=(6,4))
sns.histplot(df['play_count'], bins=50, kde=True, color='skyblue')
plt.xscale('log')
plt.title('播放量分布（对数）')
plt.xlabel('play_count')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'play_dist.png'), dpi=300)
plt.close()

# 4. 图2：点赞量分布（对数坐标）
plt.figure(figsize=(6,4))
sns.histplot(df['like_count'], bins=50, kde=True, color='salmon')
plt.xscale('log')
plt.title('点赞量分布（对数）')
plt.xlabel('like_count')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'like_dist.png'), dpi=300)
plt.close()

# 5. 图3：播放-点赞散点 + 回归线
plt.figure(figsize=(6,4))
sns.regplot(x='play_count', y='like_count', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xscale('log')
plt.yscale('log')
plt.title('播放量 vs 点赞量')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'play_like_scatter.png'), dpi=300)
plt.close()

# 6. 图4：点赞率（like/play）分布
df['like_rate'] = (df['like_count'] / df['play_count']).clip(0, 1)
plt.figure(figsize=(6,4))
sns.histplot(df['like_rate'], bins=50, kde=True, color='seagreen')
plt.title('点赞率分布')
plt.xlabel('like_rate')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'like_rate_dist.png'), dpi=300)
plt.close()

# 7. Plotly 交互仪表盘
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('播放量分布', '点赞量分布', '播放-点赞散点', '点赞率分布'),
    specs=[[{'type':'histogram'}, {'type':'histogram'}],
           [{'type':'scatter'}, {'type':'histogram'}]]
)

# 播放
fig.add_trace(go.Histogram(x=df['play_count'], name='play', nbinsx=50), row=1, col=1)
# 点赞
fig.add_trace(go.Histogram(x=df['like_count'], name='like', nbinsx=50), row=1, col=2)
# 散点
fig.add_trace(go.Scattergl(x=df['play_count'], y=df['like_count'],
                           mode='markers', marker=dict(size=3, opacity=0.3),
                           name='scatter'), row=2, col=1)
# 点赞率
fig.add_trace(go.Histogram(x=df['like_rate'], name='like_rate', nbinsx=50), row=2, col=2)

fig.update_xaxes(type='log', row=1, col=1)
fig.update_xaxes(type='log', row=1, col=2)
fig.update_xaxes(type='log', row=2, col=1)
fig.update_yaxes(type='log', row=2, col=1)

fig.update_layout(height=800, title='清洗结果概览（交互）')
html_path = os.path.join(out_dir, 'report.html')
fig.write_html(html_path)

print(f"\n✅ 可视化完成！")
print(f"📊 静态图已保存至: {out_dir}")
print(f"🌐 双击打开交互报告: {html_path}")


# 评估脚本
print("📊 数据质量评估")
print("="*50)

# 1. 数据完整性
total_cells = len(df) * len(df.columns)
missing_cells = df.isnull().sum().sum()
print(f"✅ 数据完整性: {(1 - missing_cells/total_cells)*100:.1f}%")

# 2. 标签平衡性（假设你有一列叫 label）
if 'label' in df.columns:
    gini = 1 - sum((df['label'].value_counts() / len(df)) ** 2)
    print(f"✅ 标签平衡性（Gini系数）: {gini:.3f}（越接近0越平衡）")
else:
    print("⚠️  未发现 ‘label’ 列，跳过标签平衡性评估")

# 3. CTR判别力（ANOVA效应量）
if 'CTR' in df.columns and 'label' in df.columns:
    import scipy.stats as stats
    groups = [group['CTR'].values for _, group in df.groupby('label')]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"✅ 标签判别力: F={f_stat:.2f}, p={p_val:.3g}")
else:
    print("⚠️  未发现 ‘CTR’ 或 ‘label’ 列，跳过 CTR 判别力评估")

# 4. 视觉特征有效性（随机抽检验证）
# 只有当你真的存了封面图并且写了 extract_visual_features 时才跑
if os.path.isdir("covers") and 'CTR' in df.columns and 'filename' in df.columns:
    import numpy as np
    random_100 = df.sample(100)
    high_ctr = random_100[random_100['CTR'] > random_100['CTR'].median()]
    low_ctr  = random_100[random_100['CTR'] <= random_100['CTR'].median()]

    def extract_visual_features(path):
        # 这里用伪代码占位，实际请替换成你的实现
        return [0]*10   # 返回一个包含亮度等10维特征的list

    high_brightness = np.array([extract_visual_features(f"covers/{f}")[6]
                                for f in high_ctr['filename']])
    low_brightness  = np.array([extract_visual_features(f"covers/{f}")[6]
                                for f in low_ctr['filename']])
    t_stat, p_val = stats.ttest_ind(high_brightness, low_brightness)
    print(f"✅ 亮度特征有效性: t={t_stat:.2f}, p={p_val:.3f}")
else:
    print("⚠️  缺少 ‘covers’ 目录或相关列，跳过视觉特征评估")

    # ---------- 调试专用 ----------
print("DataFrame 列名：", df.columns.tolist())          # 看列名到底叫啥
if 'label' in df.columns:
    vc = df['label'].value_counts()
    print("label 取值分布：\n", vc)
    print("label 占比：\n", (vc / vc.sum()).round(4))
    # 计算 Gini 系数（你的公式没错，再算一遍）
    gini = 1 - sum((vc / vc.sum()) ** 2)
    print("重新计算的 Gini：", gini)
else:
    print("⚠️ 确实没有叫 label 的列！")
# ------------------------------