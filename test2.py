#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision.io import read_image
from torch import nn
import torchvision.transforms.v2 as transformv2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm  # 导入tqdm用于显示进度条
import matplotlib.pyplot as plt # 导入matplotlib用于绘图

# ######################## 用户可配置参数 ##############################
MAX_EPOCHS = 40  # 最大训练轮次（根据需要修改）
USE_EARLY_STOP = True  # 是否启用早停机制
PATIENCE = 10  # 早停耐心值（仅在启用早停时有效）
OVERFIT_THRESHOLD = 0.1  # 训练/验证准确度差阈值

BATCH_SIZE = 256  # 批量大小（根据需要修改）
LEARNING_RATE = 5e-3  # 学习率（根据需要修改）
# ####################################################################

# 配置路径
BASE_DIR = r"C:\Users\ljy\Desktop\作业及资料\大三下\机器学习\机器学习-new\大作业数据tostudents\tostudents\\"
TRAIN_LABEL_PATH = BASE_DIR + r"trainlabel.csv"
TRAIN_IMG_DIR = BASE_DIR + r"train"
TEST_IMG_DIR = BASE_DIR + r"test"

# 数据加载
raw_data = pd.read_csv(TRAIN_LABEL_PATH)
train_data, valid_data = train_test_split(raw_data, test_size=0.1)


# 自定义数据集类
class AirplaneDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, return_id=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id']
        label = self.dataframe.iloc[idx]['label']
        filename = f"{img_id:05d}.png"
        image = read_image(f"{self.img_dir}/{filename}")

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        if self.return_id:
            return image, torch.tensor(label, dtype=torch.float32), img_id
        else:
            return image, torch.tensor(label, dtype=torch.float32)


# 图像预处理 - 为训练集添加更强的数据增强
image_transform_train = transformv2.Compose([
    transformv2.RandomHorizontalFlip(p=0.5),
    transformv2.RandomVerticalFlip(p=0.5),
    transformv2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transformv2.ToDtype(torch.float32, scale=True),
    transformv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_transform_val = transformv2.Compose([
    transformv2.ToDtype(torch.float32, scale=True),
    transformv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化数据集
train_dataset = AirplaneDataset(train_data, TRAIN_IMG_DIR, image_transform_train)
valid_dataset = AirplaneDataset(valid_data, TRAIN_IMG_DIR, image_transform_val)

# 类别平衡处理
targets = train_data['label']
class_counts = targets.value_counts()
weights = 1. / torch.tensor([class_counts[0], class_counts[1]], dtype=torch.float32)
sample_weights = weights[targets.values].squeeze()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# 模型定义（保持原样）
class AirplaneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
           # 第一个卷积块 - 增加通道数
            nn.Conv2d(3, 32, 3, padding=1),  # 增加到32通道
            nn.BatchNorm2d(32),              # 添加批归一化
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), # 添加额外卷积层
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 20x20 -> 10x10
            nn.Dropout(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # 添加额外卷积层
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 10x10 -> 5x5
            nn.Dropout(0.25),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    # 移到这里作为单独的层
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 只保留展平操作
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)      # 特征提取，输出形状: (batch_size, 128, 5, 5)
        x = self.avgpool(x)       # 全局平均池化，输出形状: (batch_size, 128, 1, 1)
        x = self.classifier(x)    # 分类器（包含Flatten），输出形状: (batch_size, 1)
        return x


# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 训练设备: {device}")
model = AirplaneDetector().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# 训练控制初始化
best_acc = 0.0
early_stop_counter = 0
training_active = True  # 新增训练状态标志

# 初始化用于绘图的数据存储
train_losses = []
train_accs = []
valid_accs = []
epochs_list = [] # 使用epochs_list避免与外部变量冲突

# 创建图表和子图
plt.ion()  # 开启交互模式
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training Overview')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')

# ###################### 修改后的训练循环 ##########################
print(f"\n🔧 训练配置：最大轮次 {MAX_EPOCHS} | 早停机制 {'启用' if USE_EARLY_STOP else '禁用'}")
for epoch in range(1, MAX_EPOCHS + 1):
    if not training_active:
        break    # 训练阶段
    model.train()
    train_correct = 0
    train_total = 0
    epoch_losses = []  # 每个epoch的损失列表
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    train_loop = tqdm(train_loader, desc=f"训练 Epoch {epoch}/{MAX_EPOCHS}", leave=False)
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # 记录每个batch的损失
        epoch_losses.append(loss.item())

        preds = torch.round(torch.sigmoid(outputs))
        batch_correct = (preds == labels.unsqueeze(1)).sum().item()
        train_correct += batch_correct
        train_total += labels.size(0)
        
        # 更新进度条信息
        train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_correct/labels.size(0):.4f}")

    train_acc = train_correct / train_total    # 验证阶段
    avg_epoch_loss = np.mean(epoch_losses)  # 计算平均损失
    model.eval()
    valid_correct = 0
    valid_total = 0
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        valid_loop = tqdm(valid_loader, desc=f"验证 Epoch {epoch}/{MAX_EPOCHS}", leave=False)
        for images, labels in valid_loop:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))
            batch_correct = (preds == labels.unsqueeze(1)).sum().item()
            valid_correct += batch_correct
            valid_total += labels.size(0)
            
            # 更新进度条信息
            valid_loop.set_postfix(acc=f"{batch_correct/labels.size(0):.4f}")

    valid_acc = valid_correct / valid_total

    # 学习率调度器
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print(f"🔄 学习率更新: {old_lr} -> {new_lr}")

    # 记录本轮数据
    epochs_list.append(epoch)
    train_losses.append(avg_epoch_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

    # 更新图表
    ax1.clear()
    ax1.plot(epochs_list, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    ax2.clear()
    ax2.plot(epochs_list, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs_list, valid_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以避免标题重叠
    plt.draw()
    plt.pause(0.1)  # 暂停一小段时间以便更新图表

    # 模型保存逻辑（始终保持最佳模型）
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ 发现新的最佳模型: {valid_acc:.4f}")
        early_stop_counter = 0  # 重置早停计数器
    elif USE_EARLY_STOP:  # 仅在启用早停时计数
        early_stop_counter += 1
        print(f"⚠️ 准确度未提升: 连续{early_stop_counter}次")

    # 过拟合检测（始终有效）
    if train_acc - valid_acc > OVERFIT_THRESHOLD:
        print(f"🚨 检测到过拟合（差值{train_acc - valid_acc:.2f}），强制停止！")
        training_active = False

    # 早停判断（仅在启用时生效）
    if USE_EARLY_STOP and early_stop_counter >= PATIENCE:
        print(f"🛑 早停触发：连续{PATIENCE}个epoch未提升")
        training_active = False

    print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | 训练准确度: {train_acc:.4f} | 验证准确度: {valid_acc:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load("best_model.pth"))


# 测试集预测（保持原样）
class TestDataset(Dataset):
    def __init__(self, start_id, end_id, img_dir, transform):
        self.img_ids = list(range(start_id, end_id + 1))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = f"{img_id:05d}.png"
        image = read_image(f"{self.img_dir}/{filename}")

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        return image, img_id


test_dataset = TestDataset(22400, 31999, TEST_IMG_DIR, image_transform_val)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
predictions = []
with torch.no_grad():
    test_loop = tqdm(test_loader, desc="测试集预测", leave=False)
    for images, img_ids in test_loop:
        images = images.to(device)
        outputs = torch.sigmoid(model(images))
        batch_preds = torch.round(outputs).cpu().numpy().astype(int)
        predictions.extend(zip(img_ids.numpy().flatten(), batch_preds.flatten()))
        test_loop.set_postfix(completed=f"{len(predictions)}/{len(test_dataset)}")

results_df = pd.DataFrame(predictions, columns=['id', 'label'])
results_df.to_csv("testlabel.csv", index=False)
print("✅ 测试集预测结果已保存至 testlabel.csv")

# 验证集预测（保持原样）
valid_dataset_with_id = AirplaneDataset(valid_data, TRAIN_IMG_DIR, image_transform_val, return_id=True)
valid_loader = DataLoader(valid_dataset_with_id, batch_size=BATCH_SIZE)

model.eval()
valid_predictions = []
valid_ids = []
with torch.no_grad():
    valid_loop = tqdm(valid_loader, desc="验证集预测", leave=False)
    for images, labels, ids in valid_loop:
        images = images.to(device)
        outputs = torch.sigmoid(model(images))
        batch_preds = torch.round(outputs).cpu().numpy().astype(int)
        valid_ids.extend(ids.numpy().flatten())
        valid_predictions.extend(batch_preds.flatten())
        valid_loop.set_postfix(completed=f"{len(valid_predictions)}/{len(valid_dataset_with_id)}")

valid_results = pd.DataFrame({
    'id': valid_ids,
    'label': valid_predictions
})
valid_results.to_csv("validlabel.csv", index=False)
# 计算验证集准确率
valid_accuracy = (valid_results['label'] == valid_data['label'].values).mean()
print(f"✅ 验证集预测结果已保存至 validlabel.csv 准确率: {valid_accuracy:.4f}")

plt.ioff()  # 关闭交互模式
plt.show()  # 显示图表