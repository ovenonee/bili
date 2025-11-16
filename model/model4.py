import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# # 加载 CSV 数据
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')  # 请替换为你的 CSV 文件名
# 假设列名是: filename, play_count, like_count, label
print("原始数据形状:", df.shape)
print(df.head())

# 加载图像和文本特征
visual_features = np.load(r'D:\bilitest\features\visual_X.npy')  # 形状: (N, D1)
text_features = np.load(r'D:\bilitest\features\text_X.npy')      # 形状: (N, D2)

assert len(df) == len(visual_features) == len(text_features), "样本数不一致！"

# 合并特征
X = np.concatenate([visual_features, text_features], axis=1)
y_raw = df[['play_count', 'like_count']].values  # 原始值

# 【关键改进】对目标变量进行对数变换
# 防止 log(0)，加一个小的常数
y_log = np.log1p(y_raw)  # log1p(x) = log(x + 1)

print("原始目标变量范围:")
print(f"播放量: {y_raw[:, 0].min():.0f} ~ {y_raw[:, 0].max():.0f}")
print(f"点赞量: {y_raw[:, 1].min():.0f} ~ {y_raw[:, 1].max():.0f}")
print("\n对数变换后范围:")
print(f"播放量: {y_log[:, 0].min():.2f} ~ {y_log[:, 0].max():.2f}")
print(f"点赞量: {y_log[:, 1].min():.2f} ~ {y_log[:, 1].max():.2f}")

# ========================
# 2. 划分训练集和测试集
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42, shuffle=True
)

# （可选）标准化特征（对树模型非必需，但对其他模型有帮助）
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# ========================
# 3. 构建并训练模型
# ========================
base_model = RandomForestRegressor(
    n_estimators=200,      # 增加树的数量
    max_depth=20,          # 允许更深的树
    min_samples_split=5,   # 防止过拟合
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

multi_output_model = MultiOutputRegressor(base_model)

print("开始训练模型...")
multi_output_model.fit(X_train, y_train)
print("✅ 训练完成！")

# ========================
# 4. 在测试集上评估
# ========================
y_pred_log = multi_output_model.predict(X_test)

# 【关键】将预测值反变换回原始尺度
y_pred_raw = np.expm1(y_pred_log)  # expm1(x) = exp(x) - 1
y_test_raw = np.expm1(y_test)

# 计算评估指标
mse_play = mean_squared_error(y_test_raw[:, 0], y_pred_raw[:, 0])
mse_like = mean_squared_error(y_test_raw[:, 1], y_pred_raw[:, 1])

mae_play = mean_absolute_error(y_test_raw[:, 0], y_pred_raw[:, 0])
mae_like = mean_absolute_error(y_test_raw[:, 1], y_pred_raw[:, 1])

# R² 需要在原始尺度上计算
r2_play = r2_score(y_test_raw[:, 0], y_pred_raw[:, 0])
r2_like = r2_score(y_test_raw[:, 1], y_pred_raw[:, 1])

print("\n=== 模型评估结果 (原始尺度) ===")
print(f"播放量 MSE: {mse_play:.2f}, MAE: {mae_play:.2f}, R²: {r2_play:.4f}")
print(f"点赞量 MSE: {mse_like:.2f}, MAE: {mae_like:.2f}, R²: {r2_like:.4f}")

# 打印前5个样本
print("\n=== 前5个样本预测 vs 真实值 (原始尺度) ===")
for i in range(5):
    print(f"样本 {i+1}: "
          f"播放量真值={y_test_raw[i,0]:.0f}, 预测={y_pred_raw[i,0]:.0f} | "
          f"点赞量真值={y_test_raw[i,1]:.0f}, 预测={y_pred_raw[i,1]:.0f}")

# ========================
# 5. 保存模型
# ========================
joblib.dump(multi_output_model, 'bilibili_predictor_log.pkl')
print("\n✅ 模型已保存为 'bilibili_predictor_log.pkl'")

# ========================
# 6. 可视化预测结果
# ========================
import matplotlib.pyplot as plt
# 解决中文字体问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 播放量
axes[0].scatter(y_test_raw[:, 0], y_pred_raw[:, 0], alpha=0.7)
axes[0].plot([y_test_raw[:, 0].min(), y_test_raw[:, 0].max()],
             [y_test_raw[:, 0].min(), y_test_raw[:, 0].max()], 'k--', lw=2)
axes[0].set_xlabel('真实播放量')
axes[0].set_ylabel('预测播放量')
axes[0].set_title(f'播放量预测 (R²={r2_play:.4f})')

# 点赞量
axes[1].scatter(y_test_raw[:, 1], y_pred_raw[:, 1], alpha=0.7, color='orange')
axes[1].plot([y_test_raw[:, 1].min(), y_test_raw[:, 1].max()],
             [y_test_raw[:, 1].min(), y_test_raw[:, 1].max()], 'k--', lw=2)
axes[1].set_xlabel('真实点赞量')
axes[1].set_ylabel('预测点赞量')
axes[1].set_title(f'点赞量预测 (R²={r2_like:.4f})')

plt.tight_layout()
plt.show()

# （可选）绘制残差图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
residuals_play = y_test_raw[:, 0] - y_pred_raw[:, 0]
residuals_like = y_test_raw[:, 1] - y_pred_raw[:, 1]

axes[0].scatter(y_pred_raw[:, 0], residuals_play, alpha=0.7)
axes[0].axhline(0, color='k', linestyle='--')
axes[0].set_xlabel('预测播放量')
axes[0].set_ylabel('残差 (真实 - 预测)')
axes[0].set_title('播放量残差图')

axes[1].scatter(y_pred_raw[:, 1], residuals_like, alpha=0.7, color='orange')
axes[1].axhline(0, color='k', linestyle='--')
axes[1].set_xlabel('预测点赞量')
axes[1].set_ylabel('残差 (真实 - 预测)')
axes[1].set_title('点赞量残差图')

plt.tight_layout()
plt.show()