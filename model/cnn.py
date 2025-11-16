#!/usr/bin/env python
# coding: utf-8

# In[31]:


# ==================== 单元格1：强制NVIDIA GPU ====================
import sys
import os
sys.stdout.flush()
os.environ['PYTHONUNBUFFERED'] = '1'

from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
print(f"PyTorch版本: {torch.__version__}")

# ✅ 关键修复：列出所有GPU，强制选NVIDIA
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"\n检测到 {gpu_count} 个CUDA设备:")

    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        print(f"  cuda:{i}: {name}")

        # 自动选择NVIDIA（名称包含"GeForce"或"RTX"）
        if "GeForce" in name or "RTX" in name:
            device = torch.device(f"cuda:{i}")
            print(f"\n✅ 已选择NVIDIA GPU: cuda:{i}")
            break
    else:
        # 如果没找到NVIDIA，用第一个
        device = torch.device("cuda:0")
        print(f"\n⚠️ 未识别NVIDIA，默认使用: cuda:0")
else:
    device = torch.device("cpu")
    print(f"\n❌ 未检测到CUDA，使用CPU")

# 打印最终设备
print(f"\n最终设备: {device}")
print(f"设备名称: {torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}")
# ==================== 单元格1末尾添加 ====================
# 验证导入是否成功
print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ 设备: {device}")

# 验证transforms是否可用
try:
    _test = transforms.Compose([transforms.ToTensor()])
    print("✅ transforms导入成功")
except Exception as e:
    print(f"❌ transforms导入失败: {e}")

print("\n📝 请确认以上3行都显示✅后再执行单元格2")


# In[23]:


# ==================== 单元格2：训练配置 ====================
# 关键修改：所有配置参数集中到一处，transform定义在全局作用域

# 路径配置（根据你的实际路径修改）
CSV_PATH = r"D:\bilitest\merged_data\merged_data.csv"
IMAGE_PATH = r"D:\bilitest\merged_data\covers"
MODEL_SAVE_PATH = r"D:\bilitest"

# 超参数
batch_size = 32
EPOCHS = 10
LEARNING_RATE = 0.01

# 归一化参数
MAX_PLAY = 100_000_000
MAX_LIKE = 100_000_000

# 数据预处理（关键：必须在Dataset类外部定义）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 反归一化（可视化用）
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

print("✅ 配置加载完成")


# In[24]:


# ==================== 单元格3：数据集类定义 ====================
# 关键修改：显式转换RGB格式，修复图片尺寸不一致导致的stack错误

class VideoCoverDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, is_train=True):
        """初始化数据集"""
        # 加载CSV
        try:
            self.data = pd.read_csv(csv_path)
            print(f"✅ CSV加载成功: {len(self.data)} 行")
        except Exception as e:
            print(f"❌ CSV加载失败: {e}")
            raise

        # 划分训练/测试集
        train_size = int(0.8 * len(self.data))
        if is_train:
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size:]

        # 检查必要列
        required_cols = ['filename', 'play_count', 'like_count']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列: {missing_cols}")

        self.img_dir = img_dir
        self.transform = transform

        print(f"📊 {'训练' if is_train else '测试'}集大小: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """加载单条数据"""
        row = self.data.iloc[idx]

        # 关键修改：强制转为RGB三通道，避免模式不一致
        img_path = os.path.join(self.img_dir, str(row['filename']))
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"🔥 图片加载失败: {img_path} - 错误: {e}")
            # 返回黑色占位图
            image = Image.new('RGB', (256, 256), color='black')

        # 应用transform（Resize+ToTensor+Normalize）
        if self.transform:
            image = self.transform(image)

        # 标签归一化（对数变换）
        play = np.log1p(row['play_count']) / np.log1p(MAX_PLAY)
        like = np.log1p(row['like_count']) / np.log1p(MAX_LIKE)
        target = torch.tensor([play, like], dtype=torch.float32)

        return image, target


# In[25]:


# ==================== 单元格4：数据加载验证 ====================
# 关键修改：num_workers=0（Windows兼容），打印loader信息确认数据不为空

print("📂 正在加载数据...")
from torch.utils.data import DataLoader
# 实例化数据集（传入transform）
train_dataset = VideoCoverDataset(CSV_PATH, IMAGE_PATH, transform=transform, is_train=True)
test_dataset = VideoCoverDataset(CSV_PATH, IMAGE_PATH, transform=transform, is_train=False)

# 创建DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,  # 关键：Windows下必须设为0
    pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)

# 验证数据加载
print(f"\n✅ 数据加载完成！")
print(f"   训练集: {len(train_dataset)} 条 → {len(train_loader)} 个batch")
print(f"   测试集: {len(test_dataset)} 条 → {len(test_loader)} 个batch")

# 测试读取第一个batch
print("\n🔍 测试读取第一个batch...")
for imgs, targets in train_loader:
    print(f"   图片形状: {imgs.shape} (batch, 通道, 高, 宽)")
    print(f"   标签形状: {targets.shape} (batch, 2)")
    break  # 只测试第一个batch


# In[32]:


# ==================== 单元格5：模型定义 ====================
# 关键修改：删除动态计算维度，直接硬编码338272（避免初始化时引用未定义的net）

class InceptionA(torch.nn.Module):
    """Inception模块"""
    def __init__(self, in_channel):
        super().__init__()
        # 分支1：平均池化 + 1x1卷积
        self.branch_pool = torch.nn.Conv2d(in_channel, 24, kernel_size=1)

        # 分支2：1x1卷积
        self.branch1x1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)

        # 分支3：5x5卷积 (1x1降维 → 5x5卷积)
        self.branch5x5_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 分支4：3x3卷积 (1x1降维 → 3x3卷积 → 3x3卷积)
        self.branch3x3_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        # 并行处理4个分支
        branch_pool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, padding=1, stride=1))
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))

        # 在通道维度拼接
        outputs = [branch_pool, branch1x1, branch3x3, branch5x5]
        return torch.cat(outputs, dim=1)  # 输出通道: 24+16+24+24 = 88

class Net(torch.nn.Module):
    """主网络"""
    def __init__(self):
        super().__init__()
        # 特征提取
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.inception1 = InceptionA(in_channel=10)

        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        self.inception2 = InceptionA(in_channel=20)

        self.pooling = torch.nn.MaxPool2d(2)

        # ✅ 关键修改：硬编码维度
        # 计算过程: 256 → 126 → 62, 通道数88
        # 62 * 62 * 88 = 338,272
        self.fully_connection = torch.nn.Linear(327448, 2)

    def forward(self, x):
        batch_size = x.size(0)

        # 特征提取
        x = F.relu(self.pooling(self.conv1(x)))   # 256x256 → 126x126
        x = self.inception1(x)                    # 通道: 10 → 88

        x = F.relu(self.pooling(self.conv2(x)))   # 126x126 → 62x62
        x = self.inception2(x)                    # 通道: 20 → 88

        # 展平并全连接
        x = x.view(batch_size, -1)
        x = self.fully_connection(x)
        return x

# 实例化模型
net = Net().to(device)
print(f"✅ 模型创建成功！参数量: {sum(p.numel() for p in net.parameters()):,}")


# In[33]:


# ==================== 单元格6：训练准备 ====================
# 关键修改：移除try-except，让错误暴露；显式打印组件状态

# 损失函数：均方误差（回归任务）
criterion = torch.nn.MSELoss()

# 优化器：带动量的SGD
optimizer = optim.SGD(
    net.parameters(), 
    lr=LEARNING_RATE, 
    momentum=0.5
)

# 学习率调度器：每15个epoch学习率减半
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

print("✅ 训练组件初始化完成:")
print(f"   损失函数: {criterion}")
print(f"   优化器: SGD (lr={LEARNING_RATE}, momentum=0.5)")
print(f"   学习率调度: StepLR (step=15, gamma=0.5)")


# In[34]:


# ==================== 单元格7：工具函数 ====================
# 关键修改：简化计算逻辑，添加详细的进度打印

def denormalize_targets(normalized_targets):
    """将归一化的标签还原到原始尺度"""
    play = np.expm1(normalized_targets[:, 0].cpu().numpy() * np.log1p(MAX_PLAY))
    like = np.expm1(normalized_targets[:, 1].cpu().numpy() * np.log1p(MAX_LIKE))
    return np.column_stack([play, like])

train_losses = []  # 记录训练损失
test_losses = []   # 记录测试损失

def train(epoch):
    """训练一个epoch"""
    net.train()
    loss_aver = 0.0
    total_batches = len(train_loader)

    print(f"\n🚀 Epoch {epoch+1} 训练开始 (共 {total_batches} 个batch)")

    for batch_index, (inputs, labels) in enumerate(train_loader):
        # 数据移至GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        loss_aver += loss.item()

        # 每50个batch打印一次
        if (batch_index + 1) % 50 == 0:
            print(f'   [{batch_index+1:05d}/{total_batches}] loss: {loss_aver/50:.3f}')
            loss_aver = 0.0

    # 返回平均损失
    return loss_aver / total_batches

def test(epoch):
    """测试一个epoch"""
    net.eval()
    total_mse, total_mae, total_samples = 0.0, 0.0, 0

    print(f"📊 Epoch {epoch+1} 测试开始...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            # 累积误差
            total_mse += F.mse_loss(outputs, labels, reduction='sum').item()
            total_mae += F.l1_loss(outputs, labels, reduction='sum').item()
            total_samples += labels.size(0)

    # 计算平均误差
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    # 打印结果
    print(f"   MSE: {avg_mse:.5f} | MAE: {avg_mae:.5f}")

    return avg_mse

def plot_loss_curve():
    """绘制损失曲线"""
    plt.figure(figsize=(12, 5))

    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Testing MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig('./loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📈 损失曲线已保存至 loss_curve.png")


# In[35]:


# ==================== 单元格8：主训练循环 ====================
# 关键修改：删除 if __name__ == '__main__'，直接执行；添加最佳模型保存

print("="*60)
print("🎯 开始训练...")
print(f"   Epochs: {EPOCHS} | Batch Size: {batch_size} | Device: {device}")
print("="*60)

best_loss = float('inf')  # 最佳测试损失
best_epoch = 0            # 最佳epoch

# 训练循环
for epoch in range(EPOCHS):
    # 训练
    train_loss = train(epoch)

    # 测试
    test_loss = test(epoch)

    # 记录损失
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # 学习率调度
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"   学习率已调整为: {current_lr:.6f}")

    # 保存最佳模型
    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, MODEL_SAVE_PATH)
        print(f"   💾 保存最佳模型 (Epoch {epoch+1}, Loss: {best_loss:.5f})")

    # 打印分隔线
    print("-"*60)

# 训练结束
print("\n" + "="*60)
print(f"🏁 训练完成！最佳Epoch: {best_epoch+1}, 最佳Loss: {best_loss:.5f}")
print("="*60)

# 绘制损失曲线
plot_loss_curve()


# In[ ]:




