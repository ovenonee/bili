import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ========================
# 1. 加载数据
# ========================

# 加载 CSV 数据
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')  # 请替换为你的 CSV 文件名
# 假设列名是: filename, play_count, like_count, label
print("原始数据形状:", df.shape)
print(df.head())

# 加载图像和文本特征
visual_features = np.load(r'D:\bilitest\features\visual_X.npy')  # 形状: (N, D1)
text_features = np.load(r'D:\bilitest\features\text_X.npy')      # 形状: (N, D2)

# 检查样本数是否一致
assert len(df) == len(visual_features) == len(text_features), "样本数不一致！"

# 合并特征：横向拼接图像+文本特征
X = np.concatenate([visual_features, text_features], axis=1)
y = df[['play_count', 'like_count']].values  # 目标变量，形状: (N, 2)

print("合并后特征矩阵 X 形状:", X.shape)
print("目标变量 y 形状:", y.shape)

# ========================
# 2. 划分训练集和测试集
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ========================
# 3. 构建并训练多输出回归模型
# ========================

# 使用随机森林作为基础回归器
base_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 包装成多输出回归器
multi_output_model = MultiOutputRegressor(base_model)

print("开始训练模型...")
multi_output_model.fit(X_train, y_train)
print("✅ 训练完成！")

# ========================
# 4. 在测试集上评估
# ========================

y_pred = multi_output_model.predict(X_test)

# 分别计算播放量和点赞量的评估指标
mse_play = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_like = mean_squared_error(y_test[:, 1], y_pred[:, 1])

r2_play = r2_score(y_test[:, 0], y_pred[:, 0])
r2_like = r2_score(y_test[:, 1], y_pred[:, 1])

print("\n=== 模型评估结果 ===")
print(f"播放量 MSE: {mse_play:.2f}, R²: {r2_play:.4f}")
print(f"点赞量 MSE: {mse_like:.2f}, R²: {r2_like:.4f}")

# 可选：打印前5个预测值与真实值对比
print("\n=== 前5个样本预测 vs 真实值 ===")
for i in range(5):
    print(f"样本 {i+1}: "
          f"播放量真值={y_test[i,0]:.0f}, 预测={y_pred[i,0]:.0f} | "
          f"点赞量真值={y_test[i,1]:.0f}, 预测={y_pred[i,1]:.0f}")

# ========================
# 5. 保存模型（可选）
# ========================

joblib.dump(multi_output_model, 'bilibili_predictor.pkl')
print("\n✅ 模型已保存为 'bilibili_predictor.pkl'")

# ========================
# 6. （进阶）可视化预测结果（可选）
# ========================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 播放量
axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7)
axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()],
             [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
axes[0].set_xlabel('真实播放量')
axes[0].set_ylabel('预测播放量')
axes[0].set_title(f'播放量预测 (R²={r2_play:.4f})')

# 点赞量
axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7, color='orange')
axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()],
             [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
axes[1].set_xlabel('真实点赞量')
axes[1].set_ylabel('预测点赞量')
axes[1].set_title(f'点赞量预测 (R²={r2_like:.4f})')

plt.tight_layout()
plt.show()