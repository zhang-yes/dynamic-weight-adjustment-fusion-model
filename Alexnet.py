import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import time


# 数据预处理
class ImageDataset(Dataset):
    def __init__(self, class_labels, image_paths, transform=None):
        self.class_labels = torch.tensor(class_labels, dtype=torch.long)
        self.image_paths = image_paths  # 保持为列表或pandas Series
        self.transform = transform

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # 确保路径是字符串
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_labels[idx]
        return image, label


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_src = pd.read_excel("./source.xlsx")
image_paths = df_src.iloc[:,1].tolist()
class_labels = df_src.iloc[:, 0].astype(int).values - 1

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset(class_labels, image_paths, transform)

# 计算每个集合的大小
total_size = len(dataset)
test_size = total_size // 10
val_size = total_size // 10
train_size = total_size - test_size - val_size

# 使用random_split进行划分
train_dataset, test_dataset = random_split(dataset, [train_size+val_size,test_size])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载AlexNet模型
model = models.alexnet(weights=None)  # 使用weights=None代替pretrained=False

# 修改最后的全连接层以匹配类别数
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 3)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=7e-7)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 150
start = time.time()
# 训练模型
for epoch in range(num_epochs):
    train_data, val_dataset = random_split(train_dataset, [train_size, val_size])
    # 加载数据集
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    # 创建文件夹，如果不存在，则创建
    if not os.path.exists('record'):
        os.makedirs('record')
        # 保存模型的损失
        open('./record/loss.txt', 'a').write(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}\n')



    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch + 1}, Val Accuracy: {100 * correct / total}')
    # 创建文件夹，如果不存在，则创建
    if not os.path.exists('record'):
        os.makedirs('record')
        # 保存模型的损失
        open('./record/Val Accuracy.txt', 'a').write(f'Epoch {epoch + 1}, Val Accuracy: {100 * correct / total}\n')

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch + 1}, Test Accuracy: {100 * correct / total}')
    # 创建文件夹，如果不存在，则创建
    if not os.path.exists('record'):
        os.makedirs('record')
        # 保存模型的损失
        open('./record/Test Accuracy.txt', 'a').write(f'Epoch {epoch + 1}, Test Accuracy: {100 * correct / total}\n')

end = time.time()
print(f"训练时间：{end - start}秒")