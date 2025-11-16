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
# 2. 创建二分类标签：低热度 vs 高热度
# ========================
play_counts = df['play_count'].values

# 只区分低热度和高热度，忽略中等热度
q50 = np.percentile(play_counts, 50)  # 中位数作为分界

def classify_hotness_binary(pc):
    if pc < q50:
        return '低热度'
    else:
        return '高热度'

# 只选择低热度和高热度的样本
mask = (play_counts < q50) | (play_counts > q50)  # 排除中位数附近的样本
X_filtered = X[mask]
play_filtered = play_counts[mask]
like_filtered = df['like_count'].values[mask]

hotness_labels = np.array([classify_hotness_binary(pc) for pc in play_filtered])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotness_labels)

print(f"二分类划分阈值: {q50:.0f}")
print(f"热度分布: {np.unique(hotness_labels, return_counts=True)}")
print(f"过滤后样本数: {len(X_filtered)} (原: {len(X)})")

# ========================
# 3. 特征工程
# ========================
ratio_log = np.log1p(like_filtered / (play_filtered + 1e-8))
X_enhanced = np.column_stack([X_filtered, ratio_log])

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
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=1
    )
    
    classifier.fit(X_train, y_train)
    model_name = 'XGBoost_Binary'
    
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
    model_name = 'RandomForest_Binary'

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
# 8. 保存模型
# ========================
joblib.dump(classifier, f'bilibili_classifier_{model_name.lower()}.pkl')
joblib.dump(label_encoder, 'label_encoder_binary.pkl')
joblib.dump(scaler, 'feature_scaler_binary.pkl')