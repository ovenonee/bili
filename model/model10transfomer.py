import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================
# 1. 自定义数据集类
# ========================
class BiliDataset(Dataset):
    def __init__(self, visual_features, text_features, labels):
        self.visual_features = torch.FloatTensor(visual_features)
        self.text_features = torch.FloatTensor(text_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.visual_features[idx], self.text_features[idx], self.labels[idx]

# ========================
# 2. 多模态Transformer模型（改进版）
# ========================
class MultiModalTransformer(nn.Module):
    def __init__(self, visual_dim, text_dim, num_classes, d_model=256, nhead=8, num_layers=2):
        super(MultiModalTransformer, self).__init__()
        
        # 特征投影层
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(2, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=0.3,  # 增加dropout防止过拟合
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头（增加复杂度）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, visual_features, text_features):
        # 投影到统一维度
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)
        text_proj = self.text_proj(text_features).unsqueeze(1)
        
        # 拼接两个模态
        combined = torch.cat([visual_proj, text_proj], dim=1)
        
        # 添加位置编码
        combined = combined + self.pos_encoding.unsqueeze(0)
        
        # Transformer处理
        output = self.transformer(combined)
        
        # 使用注意力池化而不是平均
        attention_weights = torch.softmax(output.mean(dim=-1), dim=1).unsqueeze(-1)
        fused_features = (output * attention_weights).sum(dim=1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits

# ========================
# 3. 加载数据
# ========================
df = pd.read_csv(r'D:\bilitest\cleaned_data\result_no0.csv')  # 请替换为你的 CSV 文件名
# 假设列名是: filename, play_count, like_count, label
print("原始数据形状:", df.shape)
print(df.head())

# 加载图像和文本特征
visual_features = np.load(r'D:\bilitest\features\visual_X.npy')  # 形状: (N, D1)
text_features = np.load(r'D:\bilitest\features\text_X.npy')      # 形状: (N, D2)

assert len(df) == len(visual_features) == len(text_features), "样本数不一致！"

# 创建分类标签
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
# 4. 数据预处理
# ========================
scaler_visual = StandardScaler()
scaler_text = StandardScaler()

visual_scaled = scaler_visual.fit_transform(visual_features)
text_scaled = scaler_text.fit_transform(text_features)

# 计算类别权重（解决不平衡问题）
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"类别权重: {class_weights.cpu().numpy()}")

# 划分数据
X_visual_train, X_visual_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    visual_scaled, text_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# 创建数据集和数据加载器
train_dataset = BiliDataset(X_visual_train, X_text_train, y_train)
test_dataset = BiliDataset(X_visual_test, X_text_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
print(f"视觉特征维度: {visual_scaled.shape[1]}, 文本特征维度: {text_scaled.shape[1]}")

# ========================
# 5. 初始化模型和优化器
# ========================
model = MultiModalTransformer(
    visual_dim=visual_scaled.shape[1],
    text_dim=text_scaled.shape[1],
    num_classes=len(np.unique(y)),
    d_model=256,
    nhead=8,
    num_layers=3  # 增加层数
).to(device)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 使用加权损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# ========================
# 6. 训练和评估函数
# ========================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for visual_batch, text_batch, labels in dataloader:
        visual_batch = visual_batch.to(device)
        text_batch = text_batch.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(visual_batch, text_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for visual_batch, text_batch, labels in dataloader:
            visual_batch = visual_batch.to(device)
            text_batch = text_batch.to(device)
            labels = labels.to(device)
            
            outputs = model(visual_batch, text_batch)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

# ========================
# 7. 训练模型
# ========================
num_epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("🚀 开始训练改进版多模态Transformer...")
best_val_f1 = 0
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
    
    # 计算F1分数用于早停
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step(val_f1)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
    
    # 早停（基于F1分数）
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_visual': scaler_visual,
            'scaler_text': scaler_text,
            'label_encoder': label_encoder,
            'optimizer_state_dict': optimizer.state_dict()
        }, 'multimodal_transformer_best_improved.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"✅ 训练完成！最佳验证F1分数: {best_val_f1:.4f}")

# ========================
# 8. 最终评估
# ========================
checkpoint = torch.load('multimodal_transformer_best_improved.pth')
model.load_state_dict(checkpoint['model_state_dict'])

final_val_loss, final_val_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
final_f1_macro = f1_score(all_labels, all_preds, average='macro')

print(f"\n=== 最终模型评估结果 ===")
print(f"准确率: {final_val_acc:.4f}")
print(f"F1宏平均: {final_f1_macro:.4f}")

target_names = label_encoder.classes_
print("\n=== 详细分类报告 ===")
print(classification_report(all_labels, all_preds, target_names=target_names))

cm = confusion_matrix(all_labels, all_preds)
print("\n=== 混淆矩阵 ===")
print(cm)

# ========================
# 9. 可视化结果
# ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='训练损失', marker='o')
ax1.plot(val_losses, label='验证损失', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('损失')
ax1.set_title('训练过程 - 损失曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accuracies, label='训练准确率', marker='o')
ax2.plot(val_accuracies, label='验证准确率', marker='s')
ax2.set_ylabel('准确率 (%)')
ax2.set_xlabel('Epoch')
ax2.set_title('训练过程 - 准确率曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('改进版多模态Transformer - 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# ========================
# 10. 保存模型
# ========================
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_visual': scaler_visual,
    'scaler_text': scaler_text,
    'label_encoder': label_encoder,
    'model_config': {
        'visual_dim': visual_scaled.shape[1],
        'text_dim': text_scaled.shape[1],
        'num_classes': len(np.unique(y)),
        'd_model': 256,
        'nhead': 8,
        'num_layers': 3
    }
}, 'multimodal_transformer_improved_final.pth')

print(f"\n✅ 改进版模型已保存为 'multimodal_transformer_improved_final.pth'")

# ========================
# 11. 示例预测
# ========================
print("\n=== 示例预测 ===")
model.eval()
with torch.no_grad():
    for i in range(5):
        idx = np.random.randint(0, len(test_dataset))
        visual_feat, text_feat, true_label = test_dataset[idx]
        
        visual_feat = visual_feat.unsqueeze(0).to(device)
        text_feat = text_feat.unsqueeze(0).to(device)
        
        output = model(visual_feat, text_feat)
        pred_label = output.argmax(1).item()
        
        true_label_name = label_encoder.inverse_transform([true_label.item()])[0]
        pred_label_name = label_encoder.inverse_transform([pred_label])[0]
        
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        print(f"样本 {i+1}: 真实={true_label_name}, 预测={pred_label_name}, "
              f"概率=[低:{probs[0]:.2f}, 中:{probs[1]:.2f}, 高:{probs[2]:.2f}]")