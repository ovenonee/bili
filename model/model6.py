import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
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
# 2. 创建分类标签
# ========================
play_counts = df['play_count'].values
# 使用分位数划分热度等级
q25 = np.percentile(play_counts, 25)  # 25% 分位数
q75 = np.percentile(play_counts, 75)  # 75% 分位数

def classify_hotness(pc):
    if pc < q25:
        return '低热度'
    elif pc < q75:
        return '中热度'
    else:
        return '高热度'

hotness_labels = np.array([classify_hotness(pc) for pc in play_counts])

print(f"热度划分阈值: 低(<{q25:.0f}), 中({q25:.0f}~{q75:.0f}), 高(>{q75:.0f})")
print(f"热度分布: {np.unique(hotness_labels, return_counts=True)}")

# 将字符串标签转换为数字标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotness_labels)  # 0: 低热度, 1: 中热度, 2: 高热度
print(f"编码后标签: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")

# ========================
# 3. 划分训练集和测试集
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")

# ========================
# 4. 训练分类模型
# ========================
# 尝试 XGBoost 分类器（如果安装了）
try:
    from xgboost import XGBClassifier
    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='multi:softprob'  # 多分类概率输出
    )
    model_name = 'XGBoost'
    print(f"✅ 使用 {model_name} 模型")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_name = 'RandomForest'
    print(f"❌ 未安装 xgboost，使用 {model_name} 模型")

classifier = base_model

print(f"开始训练 {model_name} 分类器...")
classifier.fit(X_train, y_train)
print("✅ 训练完成！")

# ========================
# 5. 评估模型
# ========================
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)  # 获取预测概率

accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== 模型评估结果 ({model_name}) ===")
print(f"准确率: {accuracy:.4f}")

# 分类报告
target_names = label_encoder.classes_
print("\n=== 详细分类报告 ===")
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵
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

# 绘制各类别预测概率分布
plt.figure(figsize=(12, 4))
for i, class_name in enumerate(target_names):
    plt.subplot(1, 3, i+1)
    class_probs = y_pred_proba[:, i]
    plt.hist(class_probs, bins=30, alpha=0.7, color=['red', 'orange', 'green'][i])
    plt.title(f'{class_name} 预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('频次')
plt.tight_layout()
plt.show()

# ========================
# 7. 保存模型和标签编码器
# ========================
joblib.dump(classifier, f'bilibili_classifier_{model_name.lower()}.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print(f"\n✅ 模型已保存为 'bilibili_classifier_{model_name.lower()}.pkl'")
print("✅ 标签编码器已保存为 'label_encoder.pkl'")

# ========================
# 8. 示例预测
# ========================
print("\n=== 示例预测 ===")
sample_idx = np.random.choice(len(X_test), 5, replace=False)
for i, idx in enumerate(sample_idx):
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
    prob = y_pred_proba[idx]
    print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label}, "
          f"概率=[低:{prob[0]:.2f}, 中:{prob[1]:.2f}, 高:{prob[2]:.2f}]")