import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
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

# 合并特征
X = np.concatenate([visual_features, text_features], axis=1)

# 【特征工程】添加更多特征
like_counts = df['like_count'].values
play_counts = df['play_count'].values

# 1. 点赞/播放比率（对数）
ratio_log = np.log1p(like_counts / (play_counts + 1e-8))

# 2. 播放量的对数
play_log = np.log1p(play_counts)

# 3. 点赞量的对数
like_log = np.log1p(like_counts)

# 4. 播放量与点赞量的差值（对数）
diff_log = play_log - like_log

# 5. 播放量的标准化（Z-score）
play_zscore = (play_counts - play_counts.mean()) / play_counts.std()

# 合并所有特征
X_enhanced = np.column_stack([
    X,                    # 原始图像+文本特征
    ratio_log,            # 点赞/播放比
    play_log,             # 播放量对数
    like_log,             # 点赞量对数
    diff_log,             # 差值
    play_zscore           # 播放量Z-score
])

print(f"增强后特征维度: {X_enhanced.shape[1]} (从 {X.shape[1]} 增加到 {X_enhanced.shape[1]})")

# ========================
# 2. 创建分类标签
# ========================
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
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotness_labels)

print(f"热度划分阈值: 低(<{q25:.0f}), 中({q25:.0f}~{q75:.0f}), 高(>{q75:.0f})")
print(f"热度分布: {np.unique(hotness_labels, return_counts=True)}")

# ========================
# 3. 数据预处理
# ========================
# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# ========================
# 4. 模型训练（使用预设的最佳参数，避免长时间搜索）
# ========================
try:
    from xgboost import XGBClassifier
    
    print("🚀 使用预优化的 XGBoost 模型...")
    
    # 使用经验性较好的参数（避免长时间搜索）
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_jobs=1  # 设置为1，避免并行问题
    )
    
    print("开始训练 XGBoost 分类器...")
    classifier.fit(X_train, y_train)
    print("✅ 训练完成！")
    
    model_name = 'XGBoost_PreOptimized'
    
except ImportError:
    print("❌ 未安装 xgboost，使用优化的随机森林")
    from sklearn.ensemble import RandomForestClassifier
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=1  # 设置为1，避免并行问题
    )
    
    print("开始训练随机森林分类器...")
    classifier.fit(X_train, y_train)
    print("✅ 训练完成！")
    
    model_name = 'RandomForest_PreOptimized'

# ========================
# 5. 评估模型
# ========================
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n=== 模型评估结果 ({model_name}) ===")
print(f"准确率: {accuracy:.4f}")
print(f"F1宏平均: {f1_macro:.4f}")

target_names = label_encoder.classes_
print("\n=== 详细分类报告 ===")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("\n=== 混淆矩阵 ===")
print(cm)

# ========================
# 6. 可视化结果
# ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'{model_name} 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# ========================
# 7. 保存模型和预处理器
# ========================
joblib.dump(classifier, f'bilibili_classifier_{model_name.lower()}.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print(f"\n✅ 模型已保存为 'bilibili_classifier_{model_name.lower()}.pkl'")
print("✅ 标签编码器已保存为 'label_encoder.pkl'")
print("✅ 特征标准化器已保存为 'feature_scaler.pkl'")

# ========================
# 8. 示例预测和概率分析
# ========================
print("\n=== 示例预测 ===")
y_pred_proba = classifier.predict_proba(X_test)
sample_idx = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(sample_idx):
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
    prob = y_pred_proba[idx]
    print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label}, "
          f"概率=[低:{prob[0]:.2f}, 中:{prob[1]:.2f}, 高:{prob[2]:.2f}]")

# ========================
# 9. 特征重要性分析（如果模型支持）
# ========================
try:
    # XGBoost 特征重要性
    feature_importance = classifier.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]  # 前10个重要特征
    top_importance = feature_importance[top_features_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), [f'Feature_{i}' for i in top_features_idx])
    plt.xlabel('重要性')
    plt.title(f'{model_name} 前10个重要特征')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 前5个重要特征索引 ===")
    for i in range(5):
        idx = top_features_idx[-(i+1)]
        imp = top_importance[-(i+1)]
        print(f"特征 {idx}: 重要性 = {imp:.4f}")
        
except AttributeError:
    print("当前模型不支持特征重要性分析")

# ========================
# 10. 额外评估指标
# ========================
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
print("\n=== 各类别详细指标 ===")
for i, class_name in enumerate(target_names):
    print(f"{class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

# ========================
# 11. 交叉验证评估（使用单进程）
# ========================
print("\n🔍 进行5折交叉验证评估...")
cv_scores = cross_val_score(
    classifier, 
    X_scaled, y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=1  # 单进程，避免编码问题
)

print(f"5折交叉验证F1宏平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")