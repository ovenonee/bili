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

# 合并基础特征
X = np.concatenate([visual_features, text_features], axis=1)

# ========================
# 2. 创建五分类标签
# ========================
play_counts = df['play_count'].values

# 使用分位数划分五类
q20 = np.percentile(play_counts, 20)
q40 = np.percentile(play_counts, 40)
q60 = np.percentile(play_counts, 60)
q80 = np.percentile(play_counts, 80)

def classify_hotness_5(pc):
    if pc < q20:
        return '极低热度'
    elif pc < q40:
        return '低热度'
    elif pc < q60:
        return '中热度'
    elif pc < q80:
        return '高热度'
    else:
        return '极高热度'

hotness_labels = np.array([classify_hotness_5(pc) for pc in play_counts])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotness_labels)

print(f"热度划分阈值:")
print(f"极低(<{q20:.0f}), 低({q20:.0f}~{q40:.0f}), 中({q40:.0f}~{q60:.0f}), 高({q60:.0f}~{q80:.0f}), 极高(>{q80:.0f})")
print(f"热度分布: {np.unique(hotness_labels, return_counts=True)}")

# ========================
# 3. 特征工程（添加安全特征）
# ========================
like_counts = df['like_count'].values

# 点赞/播放比率
ratio_log = np.log1p(like_counts / (play_counts + 1e-8))

# 合并特征
X_enhanced = np.column_stack([
    X,                    # 原始图像+文本特征
    ratio_log             # 比率特征
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
# 5. 训练XGBoost模型
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
    
    model_name = 'XGBoost_5Class'
    
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
    
    model_name = 'RandomForest_5Class'

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

plt.figure(figsize=(8, 6))
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
joblib.dump(label_encoder, 'label_encoder_5class.pkl')
joblib.dump(scaler, 'feature_scaler_5class.pkl')
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
    prob_str = ", ".join([f"{name}:{p:.2f}" for name, p in zip(target_names, prob)])
    print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label}, 概率=[{prob_str}]")

# ========================
# 10. 交叉验证评估
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

# ========================
# 11. 每个类别的详细指标
# ========================
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
print("\n=== 各类别详细指标 ===")
for i, class_name in enumerate(target_names):
    print(f"{class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

# ========================
# 12. 随机猜测基准
# ========================
n_classes = len(np.unique(y))
random_accuracy = 1.0 / n_classes
print(f"\n=== 基准比较 ===")
print(f"随机猜测准确率: {random_accuracy:.4f}")
print(f"模型准确率提升: {accuracy - random_accuracy:.4f}")