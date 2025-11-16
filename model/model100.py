# %% [markdown]
# # 🎯 视频封面 → 播放量 & 点赞数三分类（低 / 中 / 高）
# ✅ 专为你的数据定制：
#    - 图片目录: D:/大作业/mydata/covers/
#    - CSV: D:/大作业/mydata/result_no0.csv
#    - 列名: filename, play_count, like_count
# ✅ 已修复 Windows + CUDA 所有问题

# %%
# 🔧 第一步：设置中文字体（必须放在最前！）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# %%
# 🔧 导入库（精简依赖）
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
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# %% [markdown]
# ## 📂 1️⃣ 加载你的数据（已按你的路径配置）

# %%
# 🔑 直接使用你的实际路径
COVER_DIR = r"D:\bilitest\merged_data\covers"           # 注意：用 r"" 避免转义问题
METADATA_PATH = r"D:\bilitest\cleaned_data\result_no0.csv"

# 检查文件是否存在
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"❌ CSV 不存在: {METADATA_PATH}")
print(f"✅ 找到 CSV: {METADATA_PATH}")

df = pd.read_csv(METADATA_PATH)
print(f"📌 原始列名: {df.columns.tolist()}")
print(f"📌 总样本数: {len(df)}")

# 🔧 按你的列名映射（filename, play_count, like_count）
df = df.rename(columns={
    'play_count': 'views',
    'like_count': 'likes'
})

# 验证必要列
required = ['filename', 'views', 'likes']
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"❌ CSV 缺少列: {missing}，当前列: {df.columns.tolist()}")

print(f"✅ 列名映射成功 → {df.columns.tolist()}")
print("\n📊 前3行数据:")
print(df[['filename', 'views', 'likes']].head(3))

# %% [markdown]
# ## 📊 2️⃣ 播放量/点赞数分类（log10 + 量级分箱）

# %%
# 对数变换（避免跨度大问题）
df['log_views'] = np.log10(df['views'] + 1)
df['log_likes'] = np.log10(df['likes'] + 1)

# ✅ 量级分箱（按常见视频平台调整）
VIEWS_THRESH = [0, 5, 7, np.inf]   # 低: <10万, 中: 10万~1千万, 高: ≥1千万
LIKES_THRESH = [0, 4, 6, np.inf]   # 低: <1万, 中: 1万~100万, 高: ≥100万

df['views_class'] = pd.cut(df['log_views'], bins=VIEWS_THRESH, labels=['低', '中', '高'], include_lowest=True)
df['likes_class'] = pd.cut(df['log_likes'], bins=LIKES_THRESH, labels=['低', '中', '高'], include_lowest=True)

print("📈 播放量类别分布：")
print(df['views_class'].value_counts().sort_index())
print("\n📈 点赞数类别分布：")
print(df['likes_class'].value_counts().sort_index())

# 辅助函数：显示各类别实际范围
def show_class_range(df, class_col, value_col):
    print(f"\n🔍 {class_col} 对应 {value_col} 实际范围：")
    for cls in ['低', '中', '高']:
        subset = df[df[class_col] == cls][value_col]
        if len(subset) > 0:
            print(f"  {cls}: {subset.min():,} ~ {subset.max():,} (中位数: {subset.median():,.0f})")

show_class_range(df, 'views_class', 'views')
show_class_range(df, 'likes_class', 'likes')

# %% [markdown]
# ## 🖼️ 3️⃣ 数据预处理（路径拼接 + 损坏图片过滤）

# %%
# 构建完整路径（Windows 路径兼容）
df['file_path'] = df['filename'].apply(lambda x: os.path.join(COVER_DIR, str(x)))

# 检查图片是否存在
print("\n🔍 检查图片文件存在性...")
missing_files = []
for idx, row in df.iterrows():
    if not os.path.exists(row['file_path']):
        missing_files.append(row['filename'])

if missing_files:
    print(f"⚠️ {len(missing_files)} 张图片缺失，例如: {missing_files[:3]}")
    df = df[df['file_path'].apply(os.path.exists)]
else:
    print("✅ 所有图片文件存在")

# 编码类别
le_views = LabelEncoder()
le_likes = LabelEncoder()
df['views_label'] = le_views.fit_transform(df['views_class'])
df['likes_label'] = le_likes.fit_transform(df['likes_class'])

print("\n🔤 类别编码映射：")
print("播放量:", dict(zip(le_views.classes_, le_views.transform(le_views.classes_))))
print("点赞数:", dict(zip(le_likes.classes_, le_likes.transform(le_likes.classes_))))

# %%
# 自定义 Dataset（自动跳过损坏图片）
class CoverDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.valid_indices = []
        print("🔍 扫描有效图片中（跳过损坏/过小图片）...")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                img = Image.open(row['file_path']).convert('RGB')
                if min(img.size) >= 20:  # 至少 20x20 像素
                    self.valid_indices.append(i)
            except Exception as e:
                print(f"  跳过: {row['filename']} → {type(e).__name__}")
        print(f"✅ 有效图片: {len(self.valid_indices)} / {len(self.df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        img = Image.open(row['file_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['views_class'], row['likes_class']

# 图像变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 划分数据集（分层抽样）
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[['views_class', 'likes_class']].apply(tuple, axis=1),
    random_state=42
)

train_dataset = CoverDataset(train_df, transform=train_transform)
val_dataset = CoverDataset(val_df, transform=val_transform)

# 🔑 关键修复：Windows 下 num_workers 必须为 0！
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"\n🧮 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

# %% [markdown]
# ## 🧠 4️⃣ 轻量多任务模型（MobileNetV2）

# %%
class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_views=3, num_likes=3):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.views_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_views)
        )
        self.likes_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_likes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.views_head(x), self.likes_head(x)

model = MultiTaskMobileNet(
    num_views=len(le_views.classes_),
    num_likes=len(le_likes.classes_)
).to(device)
print("✅ 模型: MobileNetV2 (轻量高效，适合你的数据规模)")

# %% [markdown]
# ## ⚙️ 5️⃣ 训练设置（加权损失防不平衡）

# %%
# 🔑 对少数类加权（高类样本极少！）
weights_views = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['views_label']),
    y=train_df['views_label']
)
weights_likes = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['likes_label']),
    y=train_df['likes_label']
)

criterion_views = nn.CrossEntropyLoss(
    weight=torch.tensor(weights_views, dtype=torch.float).to(device)
)
criterion_likes = nn.CrossEntropyLoss(
    weight=torch.tensor(weights_likes, dtype=torch.float).to(device)
)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 12

# %%
def train_one_epoch(model, loader, crit_v, crit_l, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, v_cls, l_cls in tqdm(loader, desc="训练中"):
        imgs = imgs.to(device, non_blocking=True)
        v_labels = torch.tensor(le_views.transform(v_cls), dtype=torch.long).to(device)
        l_labels = torch.tensor(le_likes.transform(l_cls), dtype=torch.long).to(device)

        optimizer.zero_grad()
        v_out, l_out = model(imgs)
        loss_v = crit_v(v_out, v_labels)
        loss_l = crit_l(l_out, l_labels)
        loss = loss_v + loss_l

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, le_v, le_l):
    model.eval()
    v_true, v_pred = [], []
    l_true, l_pred = [], []
    
    with torch.no_grad():
        for imgs, v_cls, l_cls in tqdm(loader, desc="验证中"):
            imgs = imgs.to(device, non_blocking=True)
            v_labels = le_v.transform(v_cls)
            l_labels = le_l.transform(l_cls)

            v_out, l_out = model(imgs)
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
# ## ▶️ 6️⃣ 开始训练（进度条应正常滚动！）

# %%
print("\n🔥 开始训练（batch_size=8, num_workers=0, MobileNetV2）...")
best_avg_acc = 0.0

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, criterion_views, criterion_likes, optimizer, device)
    acc_v, acc_l, _, _, _, _ = evaluate(model, val_loader, device, le_views, le_likes)
    avg_acc = (acc_v + acc_l) / 2

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.4f} | "
          f"Views Acc: {acc_v:.2%} | Likes Acc: {acc_l:.2%} | Avg: {avg_acc:.2%}")

    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        torch.save(model.state_dict(), r"D:\bilitest\modual\best_cover_model.pth")  # 保存到你的目录
        print("   ✅ 保存最佳模型")

    scheduler.step()

# 加载最佳模型
model.load_state_dict(torch.load(r"D:\bilitest\modual\best_cover_model.pth", map_location=device))
print(f"\n🏆 最佳验证平均准确率: {best_avg_acc:.2%}")

# %% [markdown]
# ## 📈 7️⃣ 评估结果（重点关注高类召回率）

# %%
acc_v, acc_l, v_true, v_pred, l_true, l_pred = evaluate(model, val_loader, device, le_views, le_likes)

print(f"\n✅ 最终验证准确率 → 播放量: {acc_v:.2%} | 点赞数: {acc_l:.2%}")

# 高类召回率（关键指标！）
high_idx_v = list(le_views.classes_).index('高')
high_idx_l = list(le_likes.classes_).index('高')

high_recall_v = recall_score(v_true, v_pred, labels=[high_idx_v], average=None)[0]
high_recall_l = recall_score(l_true, l_pred, labels=[high_idx_l], average=None)[0]

print(f"\n🎯 高播放量召回率: {high_recall_v:.2%} | 高点赞召回率: {high_recall_l:.2%}")
print("（越高越好！说明模型能识别爆款视频）")

print("\n📊 播放量分类报告：")
print(classification_report(v_true, v_pred, target_names=le_views.classes_, digits=3))

print("\n📊 点赞数分类报告：")
print(classification_report(l_true, l_pred, target_names=le_likes.classes_, digits=3))

# %% [markdown]
# ## 🔮 8️⃣ 预测新图片（示例）

# %%
def predict_cover(image_path, model, transform, device, le_v, le_l):
    """预测单张封面"""
    model.eval()
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            v_out, l_out = model(img)
            v_pred = torch.argmax(v_out, dim=1).item()
            l_pred = torch.argmax(l_out, dim=1).item()
        
        v_class = le_v.inverse_transform([v_pred])[0]
        l_class = le_l.inverse_transform([l_pred])[0]
        return v_class, l_class
    except Exception as e:
        return f"❌ 错误: {e}", ""

# 示例预测（验证集第一张）
if len(val_df) > 0:
    sample = val_df.iloc[0]
    pred_v, pred_l = predict_cover(
        sample['file_path'], model, val_transform, device, le_views, le_likes
    )

    print(f"\n🔍 示例预测：{sample['filename']}")
    print(f"   真实 → 播放量: {sample['views_class']}, 点赞: {sample['likes_class']}")
    print(f"   预测 → 播放量: {pred_v}, 点赞: {pred_l}")

    # 可视化
    try:
        img = Image.open(sample['file_path']).convert('RGB')
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"真实: {sample['views_class']}/{sample['likes_class']}\n预测: {pred_v}/{pred_l}", fontsize=12)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"⚠️ 图片显示失败: {e}")

print("\n🎉 训练完成！模型已保存为: D:\bilitest\modual\best_cover_model.pth")