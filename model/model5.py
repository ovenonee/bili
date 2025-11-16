import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 XGBoost，如果未安装则回退到随机森林
try:
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    print("✅ 使用 XGBoost 模型")
    base_model = XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_name = 'XGBoost'
except ImportError:
    print("❌ 未安装 xgboost，回退到随机森林")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    base_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_name = 'RandomForest'

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

# 合并特征
X = np.concatenate([visual_features, text_features], axis=1)

# 【特征工程】添加播放量与点赞量的比率（对数）
# 这个比率本身可能是一个预测指标
like_to_play_ratio_log = np.log1p(df['like_count'].values / (df['play_count'].values + 1e-8))
X_enhanced = np.column_stack([X, like_to_play_ratio_log])

y_raw = df[['play_count', 'like_count']].values
y_log = np.log1p(y_raw)

print(f"增强后特征矩阵 X 形状: {X_enhanced.shape} (增加了比率特征)")
print("原始目标变量范围:")
print(f"播放量: {y_raw[:, 0].min():.0f} ~ {y_raw[:, 0].max():.0f}")
print(f"点赞量: {y_raw[:, 1].min():.0f} ~ {y_raw[:, 1].max():.0f}")

# ========================
# 2. 划分训练集和测试集
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y_log, test_size=0.2, random_state=42, shuffle=True
)

# ========================
# 3. 训练模型
# ========================
multi_output_model = MultiOutputRegressor(base_model)

print(f"开始训练 {model_name} 模型...")
multi_output_model.fit(X_train, y_train)
print("✅ 训练完成！")

# ========================
# 4. 评估
# ========================
y_pred_log = multi_output_model.predict(X_test)
y_pred_raw = np.expm1(y_pred_log)
y_test_raw = np.expm1(y_test)

r2_play = r2_score(y_test_raw[:, 0], y_pred_raw[:, 0])
r2_like = r2_score(y_test_raw[:, 1], y_pred_raw[:, 1])
mae_play = mean_absolute_error(y_test_raw[:, 0], y_pred_raw[:, 0])
mae_like = mean_absolute_error(y_test_raw[:, 1], y_pred_raw[:, 1])

print(f"\n=== 模型评估结果 ({model_name}, 原始尺度) ===")
print(f"播放量 MAE: {mae_play:.2f}, R²: {r2_play:.4f}")
print(f"点赞量 MAE: {mae_like:.2f}, R²: {r2_like:.4f}")

print("\n=== 前5个样本预测 vs 真实值 ===")
for i in range(5):
    print(f"样本 {i+1}: "
          f"播放量真值={y_test_raw[i,0]:.0f}, 预测={y_pred_raw[i,0]:.0f} | "
          f"点赞量真值={y_test_raw[i,1]:.0f}, 预测={y_pred_raw[i,1]:.0f}")

# ========================
# 5. 保存模型
# ========================
joblib.dump(multi_output_model, f'bilibili_predictor_{model_name.lower()}.pkl')
print(f"\n✅ 模型已保存为 'bilibili_predictor_{model_name.lower()}.pkl'")

# （可选）可视化
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test_raw[:, 0], y_pred_raw[:, 0], alpha=0.7)
    axes[0].plot([y_test_raw[:, 0].min(), y_test_raw[:, 0].max()],
                 [y_test_raw[:, 0].min(), y_test_raw[:, 0].max()], 'k--', lw=2)
    axes[0].set_xlabel('真实播放量')
    axes[0].set_ylabel('预测播放量')
    axes[0].set_title(f'{model_name} - 播放量预测 (R²={r2_play:.4f})')

    axes[1].scatter(y_test_raw[:, 1], y_pred_raw[:, 1], alpha=0.7, color='orange')
    axes[1].plot([y_test_raw[:, 1].min(), y_test_raw[:, 1].max()],
                 [y_test_raw[:, 1].min(), y_test_raw[:, 1].max()], 'k--', lw=2)
    axes[1].set_xlabel('真实点赞量')
    axes[1].set_ylabel('预测点赞量')
    axes[1].set_title(f'{model_name} - 点赞量预测 (R²={r2_like:.4f})')

    plt.tight_layout()
    plt.show()
except ImportError:
    print("未安装 matplotlib，跳过可视化。")