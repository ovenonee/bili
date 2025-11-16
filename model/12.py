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

X = np.concatenate([visual_features, text_features], axis=1)

# ========================
# 2. 创建强制三等分类标签
# ========================
play_counts = df['play_count'].values

# 按播放量排序，然后等分
sorted_indices = np.argsort(play_counts)
n_samples = len(sorted_indices)
n_per_class = n_samples // 3

# 分配标签：前1/3为低热度，中间1/3为中热度，后1/3为高热度
labels = np.zeros(n_samples, dtype=int)
labels[n_per_class:2*n_per_class] = 1  # 中热度
labels[2*n_per_class:] = 2             # 高热度
# 前1/3默认为低热度 (labels[0:n_per_class] = 0)

# 重新排列标签以匹配原始顺序
final_labels = np.zeros(n_samples, dtype=int)
final_labels[sorted_indices] = labels

print(f"强制三等分类:")
print(f"低热度样本数: {np.sum(final_labels == 0)}")
print(f"中热度样本数: {np.sum(final_labels == 1)}")
print(f"高热度样本数: {np.sum(final_labels == 2)}")
print(f"总样本数: {len(final_labels)}")

# 创建标签名称
label_names = ['低热度', '中热度', '高热度']
hotness_labels = np.array([label_names[i] for i in final_labels])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotness_labels)

# ========================
# 3. 特征工程
# ========================
like_counts = df['like_count'].values
ratio_log = np.log1p(like_counts / (play_counts + 1e-8))
X_enhanced = np.column_stack([X, ratio_log])

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
    
    classifier.fit(X_train, y_train)
    model_name = 'XGBoost_Equal_3Class'
    
except ImportError:
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
    classifier.fit(X_train, y_train)
    model_name = 'RandomForest_Equal_3Class'

# ========================
# 6. 评估
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
# 7. 交叉验证
# ========================
cv_scores = cross_val_score(
    classifier, X_scaled, y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro', n_jobs=1
)

print(f"\n5折交叉验证F1宏平均: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ========================
# 8. 可视化
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
# 9. 保存模型
# ========================
joblib.dump(classifier, f'bilibili_classifier_{model_name.lower()}.pkl')
joblib.dump(label_encoder, 'label_encoder_equal3.pkl')
joblib.dump(scaler, 'feature_scaler_equal3.pkl')