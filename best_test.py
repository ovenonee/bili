# %% [markdown]
# # 🎯 视频封面 → 播放量 & 点赞数三分类（平衡版）
# ✅ 结合最佳实践，避免过拟合

# %%
# 🔧 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# %%
# 🔧 导入库
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# %% [markdown]
# ## 🧠 1️⃣ 改进的Focal Loss

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# %% [markdown]
# ## 🧠 2️⃣ 优化的多任务模型（ResNet18 + 轻量融合）

# %%
class OptimizedMultiTaskModel(nn.Module):
    def __init__(self, num_views=3, num_likes=3, num_labels=32):
        super().__init__()
        
        # 使用 ResNet18（比ResNet50更轻量，适合你的数据）
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 移除最后的fc层
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_labels, 32)
        
        # 轻量融合分类头
        fusion_dim = 512 + 32  # ResNet18特征 + 标签特征
        
        self.views_head = nn.Sequential(
            nn.Dropout(0.4),  # 增加dropout防止过拟合
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_views)
        )
        
        self.likes_head = nn.Sequential(
            nn.Dropout(0.4),  # 增加dropout防止过拟合
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_likes)
        )
    
    def forward(self, x, label_ids=None):
        # 图像特征
        img_features = self.backbone(x).view(x.size(0), -1)  # [B, 512]
        
        # 标签特征
        if label_ids is not None:
            label_features = self.label_embedding(label_ids)  # [B, 32]
        else:
            label_features = torch.zeros(x.size(0), 32).to(x.device)
        
        # 融合特征
        fused_features = torch.cat([img_features, label_features], dim=1)
        
        # 分类预测
        views_pred = self.views_head(fused_features)
        likes_pred = self.likes_head(fused_features)
        
        return views_pred, likes_pred

# %% [markdown]
# ## 📂 3️⃣ 加载数据（使用你的路径）

# %%
COVER_DIR = r"D:\bilitest\merged_data\covers"
METADATA_PATH = r"D:\bilitest\cleaned_data\result_no0.csv"

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"❌ CSV 不存在: {METADATA_PATH}")

df = pd.read_csv(METADATA_PATH)
print(f"📌 原始列名: {df.columns.tolist()}")
print(f"📌 总样本数: {len(df)}")

df = df.rename(columns={
    'play_count': 'views',
    'like_count': 'likes'
})

required = ['filename', 'views', 'likes', 'label']
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"❌ CSV 缺少列: {missing}")

# %% [markdown]
# ## 📊 4️⃣ 分类标签生成

# %%
# 对数变换
df['log_views'] = np.log10(df['views'] + 1)
df['log_likes'] = np.log10(df['likes'] + 1)

# 量级分箱
VIEWS_THRESH = [0, 5, 7, np.inf]
LIKES_THRESH = [0, 4, 6, np.inf]

df['views_class'] = pd.cut(df['log_views'], bins=VIEWS_THRESH, labels=['低', '中', '高'], include_lowest=True)
df['likes_class'] = pd.cut(df['log_likes'], bins=LIKES_THRESH, labels=['低', '中', '高'], include_lowest=True)

print("📈 播放量类别分布：")
print(df['views_class'].value_counts().sort_index())

# 编码类别
le_views = LabelEncoder()
le_likes = LabelEncoder()
df['views_label'] = le_views.fit_transform(df['views_class'])
df['likes_label'] = le_likes.fit_transform(df['likes_class'])

# 标签编码
le_labels = LabelEncoder()
df['label_encoded'] = le_labels.fit_transform(df['label'])

print(f"🔤 标签种类: {len(le_labels.classes_)} 个")

# %% [markdown]
# ## 🖼️ 5️⃣ 数据预处理

# %%
df['file_path'] = df['filename'].apply(lambda x: os.path.join(COVER_DIR, str(x)))

print("🔍 检查图片文件存在性...")
df = df[df['file_path'].apply(os.path.exists)]
print(f"✅ 有效图片: {len(df)}")

class CoverDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.valid_indices = []
        print("🔍 扫描有效图片中...")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                img = Image.open(row['file_path']).convert('RGB')
                if min(img.size) >= 20:
                    self.valid_indices.append(i)
            except Exception as e:
                pass
        print(f"✅ 有效图片: {len(self.valid_indices)} / {len(self.df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        img = Image.open(row['file_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'views_class': row['views_class'],
            'likes_class': row['likes_class'],
            'views_label': row['views_label'],
            'likes_label': row['likes_label'],
            'label_id': row['label_encoded']
        }

# 强化数据增强（但不过度）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # 减轻增强强度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 划分数据集
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[['views_class', 'likes_class']].apply(tuple, axis=1),
    random_state=42
)

train_dataset = CoverDataset(train_df, transform=train_transform)
val_dataset = CoverDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # 增大batch_size
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"🧮 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

# %% [markdown]
# ## 🧠 6️⃣ 模型初始化

# %%
model = OptimizedMultiTaskModel(
    num_views=len(le_views.classes_),
    num_likes=len(le_likes.classes_),
    num_labels=len(le_labels.classes_)
).to(device)

print(f"✅ 模型: OptimizedMultiTaskModel (ResNet18 + Label Embedding)")

# %% [markdown]
# ## ⚙️ 7️⃣ 训练设置

# %%
# 使用 Focal Loss
criterion_views = FocalLoss(alpha=1, gamma=2).to(device)
criterion_likes = FocalLoss(alpha=1, gamma=2).to(device)

# 优化器：使用较小的学习率防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

EPOCHS = 15  # 减少训练轮数防止过拟合

# %% [markdown]
# ## 🏋️ 8️⃣ 训练函数

# %%
def train_one_epoch(model, loader, crit_v, crit_l, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="训练中"):
        images = batch['image'].to(device, non_blocking=True)
        label_ids = batch['label_id'].to(device, non_blocking=True)
        v_labels = batch['views_label'].to(device, non_blocking=True)
        l_labels = batch['likes_label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        
        v_out, l_out = model(images, label_ids)
        loss_v = crit_v(v_out, v_labels)
        loss_l = crit_l(l_out, l_labels)
        loss = loss_v + loss_l

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device, le_v, le_l):
    model.eval()
    v_true, v_pred = [], []
    l_true, l_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="验证中"):
            images = batch['image'].to(device, non_blocking=True)
            label_ids = batch['label_id'].to(device, non_blocking=True)
            v_labels = batch['views_label'].numpy()
            l_labels = batch['likes_label'].numpy()

            v_out, l_out = model(images, label_ids)
            _, v_p = torch.max(v_out, 1)
            _, l_p = torch.max(l_out, 1)

            v_pred.extend(v_p.cpu().numpy())
            l_pred.extend(l_p.cpu().numpy())
            v_true.extend(v_labels)
            l_true.extend(l_labels)
    
    acc_v = accuracy_score(v_true, v_pred)
    acc_l = accuracy_score(l_true, l_pred)
    return acc_v, acc_l, v_true, v_pred, l_true, l_pred

# %% [markdown]
# ## ▶️ 9️⃣ 开始训练

# %%
print("\n🔥 开始优化训练（ResNet18 + Focal Loss + Label Embedding）...")
best_avg_acc = 0.0
patience_counter = 0
max_patience = 5

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, criterion_views, criterion_likes, optimizer, device)
    acc_v, acc_l, _, _, _, _ = evaluate(model, val_loader, device, le_views, le_likes)
    avg_acc = (acc_v + acc_l) / 2

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.4f} | "
          f"Views Acc: {acc_v:.2%} | Likes Acc: {acc_l:.2%} | Avg: {avg_acc:.2%}")

    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        torch.save(model.state_dict(), r"D:\bilitest\modual\best_optimized_model.pth")
        print("   ✅ 保存最佳模型")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"   🛑 早停触发，最佳准确率: {best_avg_acc:.2%}")
            break

    # 学习率调度
    scheduler.step(avg_acc)

# 加载最佳模型
model.load_state_dict(torch.load(r"D:\bilitest\modual\best_optimized_model.pth", map_location=device))
print(f"\n🏆 最佳验证平均准确率: {best_avg_acc:.2%}")

# %% [markdown]
# ## 📈 📊 10️⃣ 最终评估

# %%
acc_v, acc_l, v_true, v_pred, l_true, l_pred = evaluate(model, val_loader, device, le_views, le_likes)

print(f"\n✅ 最终验证准确率 → 播放量: {acc_v:.2%} | 点赞数: {acc_l:.2%}")

# 高类召回率
high_idx_v = list(le_views.classes_).index('高')
high_idx_l = list(le_likes.classes_).index('高')

try:
    high_recall_v = recall_score(v_true, v_pred, labels=[high_idx_v], average=None)[0]
    high_recall_l = recall_score(l_true, l_pred, labels=[high_idx_l], average=None)[0]
    print(f"\n🎯 高播放量召回率: {high_recall_v:.2%} | 高点赞召回率: {high_recall_l:.2%}")
except:
    print(f"\n⚠️ 高类召回率计算失败")

print("\n📊 播放量分类报告：")
print(classification_report(v_true, v_pred, target_names=le_views.classes_, digits=3))

print("\n📊 点赞数分类报告：")
print(classification_report(l_true, l_pred, target_names=le_likes.classes_, digits=3))

print(f"\n🎉 训练完成！最佳准确率: {best_avg_acc:.2%}")
print("模型已保存为: D:\bilitest\modual\best_optimized_model2.pth")