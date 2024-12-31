import time
import random
from dynamic_weights import dy_weight
import warnings
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from matplotlib import pyplot as plt
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

df_src = pd.read_excel(r"./source.xlsx")

# 图片的地址
image_paths = df_src.iloc[:, 1:2]
# 类别标签
class_label = df_src.iloc[:, :1]
# 多光谱数据
multispectral_data = df_src.iloc[:, 4:]
print("-------读取数据完成-------")

# 只是更改了数据划分
test_size = 0.1
# 初始化权重
w1, w2 = 0.5, 0.5
epochs = 700
batch_size = 64
# 定义学习率
lr1 = 7e-7
lr2 = 5e-4
alpha = 20
beta = 0.35
# 定义模型的名称
model_1 = f'densenet121'
model_2 = f'bp'
model_ensemble = f'ensemble'


class ImageDataset(Dataset):
    def __init__(self, class_labels, image_paths, multispectral_datas, transform=None):
        self.class_labels = torch.tensor(class_labels, dtype=torch.long)
        self.image_paths = image_paths
        self.multispectral_datas = multispectral_datas
        self.transform = transform

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        path = self.image_paths[idx][0]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_labels[idx][0] - 1
        multispectral_data = self.multispectral_datas[idx].astype(np.float32)  # 转换为float32
        return image, torch.tensor(multispectral_data, dtype=torch.float), label


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("-------数据划分-------")
full_dataset = ImageDataset(class_label.values, image_paths.values, multispectral_data.values, transform=transform)
full_labels = class_label.values

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    full_dataset, full_labels, test_size=test_size, random_state=42, stratify=full_labels
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

# 模型1: densenet121
densenet121 = models.densenet121(pretrained=False)
densenet121.classifier = nn.Sequential(
    nn.Linear(1024, 4096, bias=True), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(4096, 512, bias=True), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(512, 128, bias=True), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 3, bias=True),nn.Softmax(dim=1)
)

# 模型2: bp神经网络
bp = nn.Sequential(
    nn.Linear(108, 216, bias=True), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(216, 128, bias=True), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 3, bias=True),nn.Softmax(dim=1)
)

# 定义损失函数
loss_func1 = nn.CrossEntropyLoss(reduction='sum')
loss_func2 = nn.CrossEntropyLoss(reduction='sum')
loss_func_ensemble = nn.CrossEntropyLoss(reduction='sum')

densenet121.to(device)
bp.to(device)
# 定义优化器
optimizer1 = torch.optim.Adam(densenet121.parameters(), lr=lr1)
optimizer2 = torch.optim.Adam(bp.parameters(), lr=lr2)

# 调度器
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.98)

# 存储损失和精度的列表
loss1_list, loss2_list, loss_ensemble_list = [], [], []
loss1_list2, loss2_list2, loss_ensemble_list2 = [], [], []
loss1_list3, loss2_list3, loss_ensemble_list3 = [], [], []
accuracy1_test_list, accuracy2_test_list, accuracy_ensemble_test_list = [], [], []
accuracy1_train_list, accuracy2_train_list, accuracy_ensemble_train_list = [], [], []
accuracy1_val_list, accuracy2_val_list, accuracy_val_list = [], [], []

# 创建记录文件,写入表头
with open(f'./record/{model_1}.csv', 'a') as f:
    f.write('Epoch, Train Loss,Val Loss,Test Loss,Train Accuracy,Val Accuracy, Test Accuracy\n')
with open(f'./record/{model_2}.csv', 'a') as f:
    f.write('Epoch, Train Loss,Val Loss,Test Loss,Train Accuracy,Val Accuracy, Test Accuracy\n')
with open(f'./record/{model_ensemble}.csv', 'a') as f:
    f.write('Epoch, Train Loss,Val Loss,Test Loss,Train Accuracy,Val Accuracy, Test Accuracy\n')
with open(f'./record/{model_1}vs{model_2}weights({alpha}).csv', 'a') as f:
    f.write(f'Epoch, {model_1}_weight,{model_2}_weight ,alpha\n')

# 训练模型
for epoch in range(epochs):
    r = random.randint(0, epochs)
    # 划分训练集和验证集，9:1比例
    train_dataset_train, train_dataset_val, _, t = train_test_split(
        train_dataset, train_labels, test_size=0.1, random_state=r, stratify=train_labels
    )

    # 创建训练和验证的DataLoader
    train_dataloader = DataLoader(train_dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset_val, batch_size=batch_size, shuffle=True)

    print('------------------------轮次分割线------------------------')
    densenet121.train()
    bp.train()

    temp_loss1 = 0.0
    temp_loss2 = 0.0
    temp_loss_ensemble = 0.0

    test_correct1 = 0.0
    test_correct2 = 0.0
    test_correct_ensemble = 0.0
    total_samples = 0

    time_start = time.time()  # 开始时间
    for images, multispectral_datas, labels in tqdm(train_dataloader, position=0):
        images = images.to(device)
        multispectral_datas = multispectral_datas.to(device)
        labels = labels.to(device)

        output1 = densenet121(images)
        output2 = bp(multispectral_datas)
        ensemble_output = w1 * output1 + w2 * output2

        loss1 = loss_func1(output1, labels)
        loss2 = loss_func2(output2, labels)

        # 反向传播和优化
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss_ensemble = loss_func_ensemble(ensemble_output, labels)

        temp_loss1 += loss1.item() * images.size(0)
        temp_loss2 += loss2.item() * multispectral_datas.size(0)
        temp_loss_ensemble += loss_ensemble.item() * images.size(0)

        _, predicted1 = torch.max(output1, 1)
        _, predicted2 = torch.max(output2, 1)
        _, predicted_ensemble = torch.max(ensemble_output, 1)

        test_correct1 += (predicted1 == labels).sum().item()
        test_correct2 += (predicted2 == labels).sum().item()
        test_correct_ensemble += (predicted_ensemble == labels).sum().item()
        total_samples += labels.size(0)

        loss1.backward()
        loss2.backward()

        optimizer1.step()
        optimizer2.step()

    temp_loss1 = temp_loss1 / total_samples
    temp_loss2 = temp_loss2 / total_samples
    temp_loss_ensemble = temp_loss_ensemble / total_samples

    train_accuracy1 = test_correct1 / total_samples
    train_accuracy2 = test_correct2 / total_samples
    train_accuracy_ensemble = test_correct_ensemble / total_samples

    accuracy1_train_list.append(train_accuracy1)
    accuracy2_train_list.append(train_accuracy2)
    accuracy_ensemble_train_list.append(train_accuracy_ensemble)

    loss1_list.append(temp_loss1)
    loss2_list.append(temp_loss2)
    loss_ensemble_list.append(temp_loss_ensemble)

    if epoch >= 450:
        scheduler1.step()

    if epoch >= 100 and epoch < 200:
        scheduler2.step()
    if epoch >= 200 and epoch < 350:
        lr2 = 1e-4
    if epoch >= 350:
        scheduler2.step()

    # 保存权重
    with open(f'./record/{model_1}vs{model_2}weights({alpha}).csv', 'a') as f:
        f.write(f'{epoch + 1},{w1},{w2},{alpha} \n')
    w1, w2 = dy_weight(temp_loss1, temp_loss2, w1, w2, alpha,beta)  # 更新权重
    # 验证模型
    densenet121.eval()
    bp.eval()
    temp3_loss1 = 0.0
    temp3_loss2 = 0.0
    temp3_loss_ensemble = 0.0
    with torch.no_grad():
        total_correct1, total_correct2, total_correct_ensemble = 0, 0, 0
        total_samples = 0
        for images, multispectral_datas, labels in tqdm(val_dataloader, position=0):
            images = images.to(device)
            multispectral_datas = multispectral_datas.to(device)
            labels = labels.to(device)

            output1 = densenet121(images)
            output2 = bp(multispectral_datas)

            loss1 = loss_func1(output1, labels)
            loss2 = loss_func2(output2, labels)

            _, predicted1 = torch.max(output1, 1)
            _, predicted2 = torch.max(output2, 1)
            ensemble_output = w1 * output1 + w2 * output2
            _, predicted_ensemble = torch.max(ensemble_output, 1)

            loss_ensemble = loss_func_ensemble(ensemble_output, labels)
            temp3_loss1 += loss1.item() * images.size(0)
            temp3_loss2 += loss2.item() * multispectral_datas.size(0)
            temp3_loss_ensemble += loss_ensemble.item() * images.size(0)
            total_correct1 += (predicted1 == labels).sum().item()
            total_correct2 += (predicted2 == labels).sum().item()
            total_correct_ensemble += (predicted_ensemble == labels).sum().item()
            total_samples += labels.size(0)

        temp3_loss1 = temp3_loss1 / total_samples
        temp3_loss2 = temp3_loss2 / total_samples
        temp3_loss_ensemble = temp3_loss_ensemble / total_samples

        accuracy1_val = total_correct1 / total_samples
        accuracy2_val = total_correct2 / total_samples
        accuracy_ensemble_val = total_correct_ensemble / total_samples

        accuracy1_val_list.append(accuracy1_val)
        accuracy2_val_list.append(accuracy2_val)
        accuracy_val_list.append(accuracy_ensemble_val)

        loss1_list3.append(temp3_loss1)
        loss2_list3.append(temp3_loss2)
        loss_ensemble_list3.append(temp3_loss_ensemble)

    temp2_loss1 = 0.0
    temp2_loss2 = 0.0
    temp2_loss_ensemble = 0.0
    with torch.no_grad():
        total_correct1, total_correct2, total_correct_ensemble = 0, 0, 0
        total_samples = 0
        for images, multispectral_datas, labels in tqdm(test_dataloader, position=0):
            images = images.to(device)
            multispectral_datas = multispectral_datas.to(device)
            labels = labels.to(device)

            output1 = densenet121(images)
            output2 = bp(multispectral_datas)

            loss1 = loss_func1(output1, labels)
            loss2 = loss_func2(output2, labels)

            _, predicted1 = torch.max(output1, 1)
            _, predicted2 = torch.max(output2, 1)
            ensemble_output = w1 * output1 + w2 * output2
            _, predicted_ensemble = torch.max(ensemble_output, 1)

            loss_ensemble = loss_func_ensemble(ensemble_output, labels)
            temp2_loss1 += loss1.item() * images.size(0)
            temp2_loss2 += loss2.item() * multispectral_datas.size(0)
            temp2_loss_ensemble += loss_ensemble.item() * images.size(0)
            total_correct1 += (predicted1 == labels).sum().item()
            total_correct2 += (predicted2 == labels).sum().item()
            total_correct_ensemble += (predicted_ensemble == labels).sum().item()
            total_samples += labels.size(0)

        temp2_loss1 = temp2_loss1 / total_samples
        temp2_loss2 = temp2_loss2 / total_samples
        temp2_loss_ensemble = temp2_loss_ensemble / total_samples

        accuracy1_test = total_correct1 / total_samples
        accuracy2_test = total_correct2 / total_samples
        accuracy_ensemble_test = total_correct_ensemble / total_samples

        accuracy1_test_list.append(accuracy1_test)
        accuracy2_test_list.append(accuracy2_test)
        accuracy_ensemble_test_list.append(accuracy_ensemble_test)

        loss1_list2.append(temp2_loss1)
        loss2_list2.append(temp2_loss2)
        loss_ensemble_list2.append(temp2_loss_ensemble)
    time_end = time.time()
    with open(f'./record/{model_1}.csv', 'a') as f:
        f.write(
            f'{epoch + 1},'
            f'{temp_loss1:.4f},{temp2_loss1:.4f},{temp3_loss1:.4f},'
            f'{train_accuracy1:.4f}, {accuracy1_val:.4f}, {accuracy1_test:.4f} '
            f'\n')
    with open(f'./record/{model_2}.csv', 'a') as f:
        f.write(
            f'{epoch + 1}, '
            f'{temp_loss2:.4f},{temp2_loss2:.4f}, {train_accuracy2:.4f},'
            f'{temp3_loss2:.4f},{accuracy2_val:.4f},{accuracy2_test:.4f} '
            f'\n')
    with open(f'./record/{model_ensemble}.csv', 'a') as f:
        f.write(
            f'{epoch + 1},'
            f' {temp_loss_ensemble:.4f}, {temp2_loss_ensemble:.4f},{temp3_loss_ensemble:.4f},'
            f'{train_accuracy_ensemble:.4f}, {accuracy_ensemble_val:.4f},{accuracy_ensemble_test:.4f} '
            f'\n')
    # 表格展示基本情况
    table = [[f"{model_1}",
              f"{temp_loss1:.4f}",
              f"{temp3_loss1:.4f}",
              f"{temp2_loss1:.4f}",
              f"{train_accuracy1:.4f}",
              f"{accuracy1_val:.4f}",
              f"{accuracy1_test:.4f}",
              f"{max(accuracy1_test_list):.4f}",
              f"{accuracy1_test_list.index(max(accuracy1_test_list)) + 1}"],
             [f"{model_2}",
              f"{temp_loss2:.4f}",
              f"{temp3_loss2:.4f}",
              f"{temp2_loss2:.4f}",
              f"{train_accuracy2:.4f}",
              f"{accuracy2_val:.4f}",
              f"{accuracy2_test:.4f}",
              f"{max(accuracy2_test_list):.4f}",
              f"{accuracy2_test_list.index(max(accuracy2_test_list)) + 1}"],
             ["Ensemble",
              f"{temp_loss_ensemble:.4f}",
              f"{temp3_loss_ensemble:.4f}",
              f"{temp2_loss_ensemble:.4f}",
              f"{train_accuracy_ensemble:.4f}",
              f"{accuracy_ensemble_val:.4f}",
              f"{accuracy_ensemble_test:.4f}",
              f"{max(accuracy_ensemble_test_list):.4f}",
              f"{accuracy_ensemble_test_list.index(max(accuracy_ensemble_test_list)) + 1}"]]
    print(tabulate(table, headers=["Model", "Train Loss","Val Loss", "Test Loss",
                                   "Train Accuracy", "Val Accuracy", "Test Accuracy",
                                   "Best Accuracy", "Best Epoch"], tablefmt="fancy_grid"))
    print(f"Epoch:[{epoch + 1}/{epochs}]  total time: {time_end - time_start}")
    print(f'{model_1}:learning_rate {optimizer1.param_groups[0]["lr"]}')
    print(f'{model_2}:learning_rate {optimizer2.param_groups[0]["lr"]}')
    print(f'{model_1}:weight {w1}  {model_2}:weight {w2} alpha:{alpha} beta:{beta}')

# 计算最高精度
max_accuracy1 = max(accuracy1_test_list)
max_accuracy2 = max(accuracy2_test_list)
max_accuracy_ensemble = max(accuracy_ensemble_test_list)

# 可视化损失和精度
plt.figure(figsize=(12, 9))
plt.subplot(2, 3, 1)
epochs = range(1, len(loss1_list) + 1)
plt.plot(epochs, loss1_list, label=f'{model_1} Train Loss')
plt.plot(epochs, loss2_list, label=f'{model_2} Train Loss')
plt.plot(epochs, loss_ensemble_list, label='Ensemble Train Loss')

plt.title('Train Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 2)
epochs = range(1, len(loss1_list) + 1)
plt.plot(epochs, loss1_list2, label=f'{model_1} Test Loss')
plt.plot(epochs, loss2_list2, label=f'{model_2} Test Loss')
plt.plot(epochs, loss_ensemble_list2, label='Ensemble Test Loss')

plt.title('Test Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(epochs, accuracy1_train_list, label=f'{model_1} Train Accuracy')
plt.plot(epochs, accuracy2_train_list, label=f'{model_2} Train Accuracy')
plt.plot(epochs, accuracy_ensemble_train_list, label='Ensemble Train Accuracy')
plt.title('Train Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(epochs, accuracy1_test_list, label=f'{model_1} Test Accuracy')
plt.plot(epochs, accuracy2_test_list, label=f'{model_2} Test Accuracy')
plt.plot(epochs, accuracy_ensemble_test_list, label='Ensemble Test Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs, loss1_list3, label=f'{model_1} Val Loss')
plt.plot(epochs, loss2_list3, label=f'{model_2} Val Loss')
plt.plot(epochs, loss_ensemble_list3, label='Ensemble Val Loss')
plt.title('Val Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs, accuracy1_val_list, label=f'{model_1} Val Accuracy')
plt.plot(epochs, accuracy2_val_list, label=f'{model_2} Val Accuracy')
plt.plot(epochs, accuracy_val_list, label='Ensemble Val Accuracy')
plt.title('Val Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig(f'./OutputGraph/{model_1}_{model_2}_{alpha}.png')
plt.show()

# 表格展示最高精度
table = [["Model 1", f"{max_accuracy1:.4f}"],
         ["Model 2", f"{max_accuracy2:.4f}"],
         ["Ensemble", f"{max_accuracy_ensemble:.4f}"]]

print(tabulate(table, headers=["Model", "Max Accuracy"], tablefmt="fancy_grid"))
