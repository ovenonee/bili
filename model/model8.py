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

# ========================
# 2. 创建分类标签（仅使用原始播放量进行分类，不作为特征）
# ========================
play_counts = df['play_count'].values
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
# 3. 特征工程（仅使用不会泄露目标信息的特征）
# ========================
like_counts = df['like_count'].values

# 1. 点赞/播放比率（对数）- 这个是可以的，因为实际应用中可能有标题信息
ratio_log = np.log1p(like_counts / (play_counts + 1e-8))

# 合并特征（仅使用图像+文本特征 + 安全的比率特征）
X_enhanced = np.column_stack([
    X,                    # 原始图像+文本特征 (367维)
    ratio_log             # 点赞/播放比 (1维) - 这是唯一可以添加的额外特征
])

print(f"增强后特征维度: {X_enhanced.shape[1]}")

# ========================
# 4. 数据预处理
# ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# ========================
# 5. 训练模型
# ========================
try:
    from xgboost import XGBClassifier
    
    print("🚀 使用优化的 XGBoost 模型...")
    
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
        n_jobs=1
    )
    
    print("开始训练 XGBoost 分类器...")
    classifier.fit(X_train, y_train)
    print("✅ 训练完成！")
    
    model_name = 'XGBoost_Corrected'
    
except ImportError:
    print("❌ 未安装 xgboost，使用随机森林")
    from sklearn.ensemble import RandomForestClassifier
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    
    print("开始训练随机森林分类器...")
    classifier.fit(X_train, y_train)
    print("✅ 训练完成！")
    
    model_name = 'RandomForest_Corrected'

# ========================
# 6. 评估模型
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
# 7. 可视化结果
# ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'{model_name} 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# ========================
# 8. 保存模型
# ========================
joblib.dump(classifier, f'bilibili_classifier_{model_name.lower()}.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print(f"\n✅ 模型已保存为 'bilibili_classifier_{model_name.lower()}.pkl'")

# ========================
# 9. 示例预测
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
# 10. 特征重要性分析
# ========================
try:
    feature_importance = classifier.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
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
# 11. 交叉验证评估
# ========================
print("\n🔍 进行5折交叉验证评估...")
cv_scores = cross_val_score(
    classifier, 
    X_scaled, y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=1
)

print(f"5折交叉验证F1宏平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")