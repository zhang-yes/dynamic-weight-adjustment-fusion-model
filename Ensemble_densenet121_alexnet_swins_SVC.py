import warnings
from collections import Counter

from prettytable import PrettyTable
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
from torchvision import transforms, models
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from var_module import var_module
from random import random
import torch
from regularization import Regularization

# TODO 1 =========================================================读取原始数据，划分数据集=========================================================
print("-------读取数据ing ~~~ -------")
df_src = pd.read_excel("./source.xlsx")
# 图片的地址
image_paths = df_src.iloc[:, 1:2]
# 类别标签
class_label = df_src.iloc[:, :1]
# 多光谱数据
multispectral_data = df_src.iloc[:, 4:]
print("-------读取数据完成-------")


class ImageDataset(Dataset):
    '''
    自定义数据集类，继承自torch.utils.data.Dataset
    '''

    def __init__(self, class_labels, image_paths, multispectral_datas, transform=None):
        self.class_labels = torch.tensor(class_labels, dtype=torch.long)
        self.image_paths = image_paths
        self.multispectral_datas = multispectral_datas
        self.transform = transform

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        try:
            path = self.image_paths[idx][0]
            # 打开图像文件
            image = Image.open(path).convert('RGB')
            if self.transform:
                # 对图像进行预处理
                image = self.transform(image)
            # 获取对应的标签,注意标签从0开始
            label = self.class_labels[idx][0] - 1
            # 获取对应的多光谱数据
            multispectral_data = self.multispectral_datas[idx]
            return image, multispectral_data, label
        except OSError as e:
            path = self.image_paths[idx][0]
            # 打印出现异常的路径和错误信息
            print(f"无法加载图像：{path}，错误：{e}")
            idx = idx + 1


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("-------/(ㄒoㄒ)/数据划分ing ~~~ -------")
# 我的数据集和标签已经准备好了
full_dataset = ImageDataset(class_label.values, image_paths.values, multispectral_data.values, transform=transform,)
full_labels = class_label.values

# 首先，将数据集分为训练集和测试集
train_dataset, test_dataset, train_labels, test_labels = train_test_split(full_dataset, full_labels, test_size=0.2,
                                                                          random_state=42, stratify=full_labels)

batch_size = 64  # 批次大小

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"(●'◡'●)训练集划分完成~~，训练集大小：{len(train_dataset)}")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"(●'◡'●)测试集划分完成~~，测试集大小：{len(test_dataset)}")
print("-------数据划分完成 ~~~ -------")
model_svc = SVC(kernel="linear",probability=True, C=1, random_state=42)

train_loss = 0
train_acc = 0
train_count = 0

for images, X_batch, y_batch in train_dataloader:
    X_batch = X_batch.numpy()
    y_batch =y_batch.numpy()

    model_svc.fit(X_batch, y_batch)
    predictions = model_svc.predict(X_batch)
    train_loss += log_loss(y_batch, model_svc.predict_proba(X_batch))
    train_acc += accuracy_score(y_batch, predictions)
    train_count += 1

train_loss /= train_count
train_acc /= train_count

# 训练 SVC 模型
train_loss = 0
train_acc = 0
train_count = 0

for images, X_batch, y_batch in train_dataloader:
    X_batch = X_batch.numpy()
    y_batch = y_batch.numpy()
    model_svc.fit(X_batch, y_batch)
    predictions = model_svc.predict(X_batch)
    train_loss += log_loss(y_batch, model_svc.predict_proba(X_batch), labels=np.unique(y_batch))
    train_acc += accuracy_score(y_batch, predictions)
    train_count += 1

train_loss /= train_count
train_acc /= train_count

# 测试 SVC 模型
test_loss = 0
test_acc = 0
test_count = 0

for images, X_batch, y_batch in test_dataloader:
    X_batch = X_batch.numpy()
    y_batch = y_batch.numpy()
    predictions = model_svc.predict(X_batch)
    test_loss += log_loss(y_batch, model_svc.predict_proba(X_batch), labels=np.unique(y_batch))
    test_acc += accuracy_score(y_batch, predictions)
    test_count += 1

test_loss /= test_count
test_acc /= test_count

print(f"SVC Training Loss: {train_loss}")
print(f"SVC Training Accuracy: {train_acc}")
print(f"SVC Testing Loss: {test_loss}")
print(f"SVC Testing Accuracy: {test_acc}")

print("-------SVC训练完成 ~~~ -------")

# TODO 2 =========================================================定义模型，加载模型，初始化模型=========================================================

print("-------(ง •_•)ง模型加载ing ~~~ -------")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device ~~ : {device}')

# 模型1:densenet121
densenet121 = models.densenet121(pretrained=False)
densenet121.classifier = nn.Sequential(
    nn.Linear(1024, 4096, bias=True),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(4096, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 128, bias=True),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 3, bias=True),
)

# 模型2:alexnet
alexnet = models.alexnet(pretrained=False, num_classes=3)  # 加载预训练的alexnet模型
# 由于AlexNet的第一个卷积层步长为4，我们可以通过调整卷积核大小来适应新的输入尺寸
# 替换AlexNet的第一个卷积层

alexnet.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=128, bias=True),
    nn.Linear(in_features=128, out_features=3, bias=True)
)

# 模型3:resnet50
resnet101 = models.swin_s(pretrained=False, num_classes=3)  # 加载预训练的resnet50模型
resnet101.fc = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=2048, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=128, bias=True),
    nn.Linear(in_features=128, out_features=3, bias=True)
)

# 将模型移动到GPU上
densenet121.to(device)
alexnet.to(device)
resnet101.to(device)

print("-------模型初始化完成 ~~~ -------")

print("-------初始化参数ing ~~~ -------")
# 损失函数
loss_func_densenet121 = nn.CrossEntropyLoss().to(device)
loss_func_alexnet = nn.CrossEntropyLoss().to(device)
loss_func_resnet50 = nn.CrossEntropyLoss().to(device)

loss_func_ensemble = nn.CrossEntropyLoss().to(device)

# 学习率
lr1 = 7e-7  #  模型 1差不多这个
lr2 = 7e-6  #  模型 2 差不多这个 7e-6
lr3 = 7e-7  #  模型 2 差不多这个
# 优化器
optimizer_densenet121 = optim.Adam(densenet121.parameters(), lr=lr1)
optimizer_alexnet = optim.Adam(alexnet.parameters(), lr=lr2)
optimizer_resnet50 = optim.Adam(resnet101.parameters(), lr=lr3)
# 调度器
scheduler_densenet121 = torch.optim.lr_scheduler.ExponentialLR(optimizer_densenet121, gamma=0.98)
scheduler_alexnet = torch.optim.lr_scheduler.ExponentialLR(optimizer_alexnet, gamma=0.98)
scheduler_resnet50 = torch.optim.lr_scheduler.ExponentialLR(optimizer_resnet50, gamma=0.98)

print("-------初始化参数完成 ~~~ -------")

# TODO 3 =========================================================训练模型，验证模型=========================================================

print("------- ( •̀ ω •́ )y 训练ing ~~~ -------")
model1_name = 'densenet121'
model2_name = 'alexnet'
model3_name = 'resnet50'
model4_name = 'svc'
model5_name = 'ensemble'


def train_module_RGB(model, loss_function, image, labels, optimizer, p=0.9, weight_decay=10e-4):
    '''
    只用RGB数据去训练深度学习模型
    :param model: 模型
    :param loss_function: 损失函数
    :param image: RGB图片
    :param labels: 标签
    :param optimizer: 优化器
    :param p: 随机丢失的p值
    :param weight_decay: 权重衰减
    :return:
    '''
    a = random()
    outputs = model(image)  # 模型的预测标签
    pre_label = torch.argmax(outputs, dim=1)
    if a <= p:
        train_loss = 0
        train_acc = 0
        # 正则优化
        reg_loss = Regularization(model, weight_decay, p=2).to(device)
        # optimizer.zero_grad()
        # 计算损失
        loss = loss_function(outputs, labels)
        # 正则优化
        loss = loss + reg_loss(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * image.size(0)
        ret, pred = torch.max(outputs.data, 1)
        correct_counts = pred.eq(labels.data.view_as(pred))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * image.size(0)
        train_data_len = len(image)
    else:
        train_loss = 0
        train_acc = 0
        train_data_len = 0
    return train_acc, train_loss, train_data_len, pre_label


def train_and_valid(model_1, model_2, model_3, model_svc,
                    loss_function_1, loss_function_2, loss_function_3,
                    optimizer_1, optimizer_2, optimizer_3,
                    epochs=100):
    history_1 = []
    history_2 = []
    history_3 = []
    history_4 = []
    history_ensemble = []

    best_acc_1 = 0.0
    best_acc_2 = 0.0
    best_acc_3 = 0.0
    best_acc_4 = 0.0
    # 组合模型最佳验证精度
    best_acc_ensemble = 0.0

    best_epoch_1 = 0
    best_epoch_2 = 0
    best_epoch_3 = 0
    best_epoch_4 = 0

    best_epoch_ensemble = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print('------------------------轮次分割线------------------------')
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        print(f"This epoch's lr is {optimizer_1.param_groups[0]['lr']}")

        path = f'./Best_model_record/'


        model_1.train()
        model_2.train()
        model_3.train()

        train_loss_1 = 0.0
        train_loss_2 = 0.0
        train_loss_3 = 0.0
        train_loss_4 = 0.0
        # train_loss_ensemble = 0.0

        train_acc_1 = 0.0
        train_acc_2 = 0.0
        train_acc_3 = 0.0
        train_acc_4 = 0.0

        # train_acc_ensembel = 0

        valid_loss_1 = 0.0
        valid_loss_2 = 0.0
        valid_loss_3 = 0.0
        valid_loss_4 = 0.0
        valid_loss_ensemble =0.0
        # valid_loss_ensemble = 0.0
        # train_acc_ensemble = 0.0

        valid_acc_1 = 0.0
        valid_acc_2 = 0.0
        valid_acc_3 = 0.0
        valid_acc_4 = 0.0
        # valid_acc_ensembel = 0

        train_data_len_1 = 0
        train_data_len_2 = 0
        train_data_len_3 = 0
        train_data_len_4 = 0

        var_data_len_1 = 0
        var_data_len_2 = 0
        var_data_len_3 = 0
        var_data_len_4 = 0

        res = 0

        for i, (image, multispectral_data, labels) in enumerate(tqdm(train_dataloader)):
            image = image.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()


            train_acc_single_1, train_loss_single_1, train_data_single_len1, output_model1_label = train_module_RGB(model_1, loss_function_1, image, labels, optimizer_1)
            train_acc_1 += train_acc_single_1
            train_loss_1 += train_loss_single_1
            train_data_len_1 += train_data_single_len1

            train_acc_single_2, train_loss_single_2, train_data_single_len2, output_model2_label = train_module_RGB(model_2, loss_function_2, image, labels, optimizer_2)
            train_acc_2 += train_acc_single_2
            train_loss_2 += train_loss_single_2
            train_data_len_2 += train_data_single_len2

            train_acc_single_3, train_loss_single_3, train_data_single_len3, output_model3_label = train_module_RGB(model_3, loss_function_3, image, labels, optimizer_3)
            train_acc_3 += train_acc_single_3
            train_loss_3 += train_loss_single_3
            train_data_len_3 += train_data_single_len3

            # 用训练好的svc对数据进行预测
            output_model4_label = torch.tensor(model_svc.predict(multispectral_data.numpy())).to(device)

            all_outputs = torch.stack((output_model1_label, output_model2_label, output_model3_label, output_model4_label))

            ensemble_label = torch.mode(all_outputs, dim=0)[0]

            # print(f'''
            # 这是真实标签:{labels},
            # 模型1输出:{output_model1_label},
            # 模型2输出:{output_model2_label},
            # 模型3输出:{output_model3_label},
            # 模型4输出:{output_model4_label},
            # ensemble输出:{ensemble_label}, ''')
            # train_loss_ensemble += loss_func_ensemble(ensemble_label, labels)

            res += (labels.cpu() == ensemble_label.cpu()).sum().item()


            # 计算准确率
            # train_acc_single_4 = accuracy_score(labels.cpu().numpy(), ensemble_label.cpu().numpy())
            # train_acc_4 += train_acc_single_4

            if epoch >= 200:
            # 改进需打开
                scheduler_densenet121.step()
                scheduler_alexnet.step()
                scheduler_resnet50.step()

        with torch.no_grad():
            num = 0
            num_classes = 3
            model_1.eval()
            model_2.eval()
            model_3.eval()

            matrix_1 = torch.zeros(num_classes, num_classes).type(torch.long)
            matrix_2 = torch.zeros(num_classes, num_classes).type(torch.long)
            matrix_3 = torch.zeros(num_classes, num_classes).type(torch.long)

            for j, (image, multispectral_data, label) in enumerate(tqdm(test_dataloader)):
                input = image.to(device)
                label = label.to(device)
                multispectral_data = multispectral_data.numpy()

                valid_loss_single_1, valid_acc_single_1, matrix_single_1, output_1, var_data_single_len1, output_normal_1, output1_label = var_module(model_1, loss_function_1, input, label, matrix_1)
                valid_loss_1 += valid_loss_single_1
                valid_acc_1 += valid_acc_single_1
                matrix_1 = matrix_single_1
                var_data_len_1 += var_data_single_len1

                valid_loss_single_2, valid_acc_single_2, matrix_single_2, output_2, var_data_single_len2, output_normal_2, output2_label = var_module(model_2, loss_function_2, input, label, matrix_2)
                valid_loss_2 += valid_loss_single_2
                valid_acc_2 += valid_acc_single_2
                matrix_2 = matrix_single_2
                var_data_len_2 += var_data_single_len2

                valid_loss_single_3, valid_acc_single_3, matrix_single_3, output_3, var_data_single_len3, output_normal_3, output3_label = var_module(model_3, loss_function_3, input, label, matrix_3)
                valid_loss_3 += valid_loss_single_3
                valid_acc_3 += valid_acc_single_3
                matrix_3 = matrix_single_3
                var_data_len_3 += var_data_single_len3

                output_model4_label = torch.tensor(model_svc.predict(multispectral_data)).to(device)

                all_outputs = torch.stack((output1_label, output2_label, output3_label, output_model4_label))
                ensemble_label = torch.mode(all_outputs, dim=0)[0]

                # print(f'''
                # 真实标签:{label},
                # 模型1输出:{output1_label},
                # 模型2输出:{output2_label},
                # 模型3输出:{output3_label},
                # 模型4输出:{output_model4_label},
                # ensemble输出:{ensemble_label},
                # ''')

                # 计算准确率

                num += (label.cpu() == ensemble_label.cpu()).sum().item()
                # ni
                # train_acc = accuracy_score(label.cpu().numpy(), ensemble_label.cpu().numpy())
                # train_acc_ensemble += train_acc * len(label)


            avg_train_loss_1 = train_loss_1 / train_data_len_1
            avg_train_acc_1 = train_acc_1 / train_data_len_1

            avg_train_loss_2 = train_loss_2 / train_data_len_2
            avg_train_acc_2 = train_acc_2 / train_data_len_2

            avg_train_loss_3 = train_loss_3 / train_data_len_3
            avg_train_acc_3 = train_acc_3 / train_data_len_3

            avg_train_loss_4 = train_loss_4 / train_data_len_1
            avg_train_acc_4 = train_acc_4 / train_data_len_1

            avg_valid_loss_1 = valid_loss_1 / var_data_len_1
            avg_valid_acc_1 = valid_acc_1 / var_data_len_1

            avg_valid_loss_2 = valid_loss_2 / var_data_len_2
            avg_valid_acc_2 = valid_acc_2 / var_data_len_2

            avg_valid_loss_3 = valid_loss_3 / var_data_len_3
            avg_valid_acc_3 = valid_acc_3 / var_data_len_3


            # avg_valid_loss_4 = valid_loss_4 / train_data_len_1
            # avg_valid_acc_4 = valid_acc_4 / train_data_len_1

            # avg_valid_loss_5 = valid_loss_ensemble / train_data_len_1
            # avg_train_loss_5 = train_loss_ensemble / train_data_len_1


            # 计算组合模型的平均验证精度与平均验证损失
            avg_train_ensembel = res / len(train_dataset)
            avg_valid_ensembel = num / len(test_dataset)

        # 将数据保存在一维列表
        list_history_1 = [epoch + 1, round(avg_train_loss_1, 4), round(avg_valid_loss_1, 4),
                          round(avg_train_acc_1, 4), round(avg_valid_acc_1, 4)]

        list_history_2 = [epoch + 1, round(avg_train_loss_2, 4), round(avg_valid_loss_2, 4),
                          round(avg_train_acc_2, 4), round(avg_valid_acc_2, 4)]

        list_history_3 = [epoch + 1, round(avg_train_loss_3, 4), round(avg_valid_loss_3, 4),
                          round(avg_train_acc_3, 4), round(avg_valid_acc_3, 4)]

        # list_history_4 = [epoch + 1, round(avg_train_loss_4, 4), round(avg_valid_loss_4, 4),
        #                   round(avg_train_acc_4, 4), round(avg_valid_acc_4, 4)]

        # list_history_5 = [epoch + 1, round(avg_train_loss_4, 4), round(avg_valid_loss_4, 4),
        #                   round(avg_train_ensembel, 4), round(avg_valid_ensembel, 4)]



        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data_1 = pd.DataFrame([list_history_1])
        data_2 = pd.DataFrame([list_history_2])
        data_3 = pd.DataFrame([list_history_3])
        # data_4 = pd.DataFrame([list_history_4])



        data_1.to_csv(f"./record/{model1_name}.csv", mode='a', header=False,
                      index=False)  # mode设为a,就可以向csv文件追加数据了
        data_2.to_csv(f"./record/{model2_name}.csv", mode='a', header=False, index=False)
        data_3.to_csv(f"./record/{model3_name}.csv", mode='a', header=False, index=False)
        # data_4.to_csv(f"./A_L_record/{model4_name}.csv", mode='a', header=False, index=False)
        # data_4.to_csv(f"./A_L_record/{model4_name}.csv", mode='a', header=False, index=False)


        history_1.append([avg_train_loss_1, avg_valid_loss_1, avg_train_acc_1, avg_valid_acc_1])
        history_2.append([avg_train_loss_2, avg_valid_loss_2, avg_train_acc_2, avg_valid_acc_2])
        history_3.append([avg_train_loss_3, avg_valid_loss_3, avg_train_acc_3, avg_valid_acc_3])
        history_4.append([train_loss, test_loss, train_acc, test_acc])
        history_ensemble.append([avg_train_ensembel, avg_valid_ensembel])

        if best_acc_1 < avg_valid_acc_1:
            best_acc_1 = avg_valid_acc_1
            best_epoch_1 = epoch + 1

        if best_acc_2 < avg_valid_acc_2:
            best_acc_2 = avg_valid_acc_2
            best_epoch_2 = epoch + 1

        if best_acc_3 < avg_valid_acc_3:
            best_acc_3 = avg_valid_acc_3
            best_epoch_3 = epoch + 1


        if best_acc_ensemble < avg_valid_ensembel:
            best_acc_ensemble = avg_valid_ensembel
            best_epoch_ensemble = epoch + 1

        epoch_end = time.time()
        time_total = round(epoch_end - epoch_start, 1)

        # 每轮结果
        table = PrettyTable(['Model', 'Train Loss', 'Var Loss', 'Train Acc', 'Var Acc'])

        table.add_row([model1_name,
                       round(avg_train_loss_1, 2), round(avg_valid_loss_1, 2),
                       round(avg_train_acc_1 * 100, 2), round(avg_valid_acc_1 * 100, 2)])

        table.add_row([model2_name,
                       round(avg_train_loss_2, 2), round(avg_valid_loss_2, 2),
                       round(avg_train_acc_2 * 100, 2), round(avg_valid_acc_2 * 100, 2)])

        table.add_row([model3_name,
                       round(avg_train_loss_3, 2), round(avg_valid_loss_3, 2),
                       round(avg_train_acc_3 * 100, 2), round(avg_valid_acc_3 * 100, 2)])

        table.add_row([model4_name,
                       round(train_loss, 2), round(test_loss, 2),
                       round(train_acc * 100, 2), round(test_acc *100,2)])

        table.add_row(['model_ensemble',
                       "", "",
                       round(avg_train_ensembel * 100, 2), round(avg_valid_ensembel * 100, 2)])

        # 验证集最佳结果
        table_2 = PrettyTable(['Model', 'Best Acc', 'Best Epoch'])
        table_2.add_row([model1_name, round(best_acc_1 * 100, 2), round(best_epoch_1, 2)])
        table_2.add_row([model2_name, round(best_acc_2 * 100, 2), round(best_epoch_2, 2)])
        table_2.add_row([model3_name, round(best_acc_3 * 100, 2), round(best_epoch_3, 2)])
        table_2.add_row([model4_name, round(test_acc * 100, 2), round(1, 2)])
        table_2.add_row(['model_ensemble', round(best_acc_ensemble * 100, 2), round(best_epoch_ensemble, 2)])

        print(table)
        print(table_2)

        print(f'Total time spent in {epoch + 1} epoch : {time_total}s')

        with open(f'./Best_model_record/log.txt', 'a', encoding='utf8') as f:
            f.write(f"{epoch + 1}\n\t"
                    f"Best Accuracy for validation:\n\t"
                    f"{model_1} : {best_acc_1:.4f} at epoch {best_epoch_1:03d}\n\t"
                    f"{model_2} : {best_acc_2:.4f} at epoch {best_epoch_2:03d}\n\t"
                    f"{model_3} : {best_acc_3:.4f} at epoch {best_epoch_3:03d}\n\t"
                    f"{model_svc} : {test_acc:.4f} at epoch {1:03d}\n\t"
                    f"model_ensemble : {best_acc_ensemble:.4f} at epoch {best_epoch_ensemble:03d}\n\n")
        # if epoch >= 70:
        #
        #     # path = "f'./Models_save/{flag}"
        #     # if not os.path.exists(path):
        #     #     os.mkdir(path)
        #     list = [model_1, model2_name, model3_name, model4_name]
        #     for name in list:
        #         path = f'./Models_save/{name}'
        #         if not os.path.exists(path):
        #             os.makedirs(path)
        #     torch.save(model_1.state_dict(), f'./Models_save/{model_1}/{model_1}_{epoch + 1}.pth')
        #     torch.save(model_2.state_dict(), f'./Models_save/{model2_name}/{model2_name}_{epoch + 1}.pth')
        #     torch.save(model_3.state_dict(), f'./Models_save/{model3_name}/{model3_name}_{epoch + 1}.pth')
        #     torch.save(model_4.state_dict(), f'./Models_save/{model4_name}/{model4_name}_{epoch + 1}.pth')
    return history_1, history_2, history_3, history_4,history_ensemble


num_epochs = 400
history_1, history_2, history_3, history_4 ,history_ensemble = train_and_valid(
    alexnet, densenet121, resnet101, model_svc,
    loss_func_alexnet, loss_func_densenet121, loss_func_resnet50,
    optimizer_alexnet, optimizer_densenet121, optimizer_resnet50,
    num_epochs
)
# torch.save(history, 'models_save/'+f'290Dataset_{model_name}_history.pth')

model1_name = 'model_1'
model2_name = 'model_2'
model3_name = 'model_3'
model4_name = 'model_4'
model5_name = 'model_ensemble'

list = [model1_name, model2_name, model3_name, model4_name, model5_name]
for name in list:
    path = f'./OutputGraph/{name}'
    if not os.path.exists(path):
        os.makedirs(path)



history = np.array(history_1)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 2.0)
plt.savefig(f"./OutputGraph/{model1_name}/loss.png")
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(f"./OutputGraph/{model1_name}/acc.png")
plt.show()

print('--------------------结果可视化~~-----------------------------------------------------')
history = np.array(history_2)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 3.0)
plt.savefig(f"./OutputGraph/{model2_name}/loss.png")
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(f"./OutputGraph/{model2_name}/acc.png")
plt.show()

history = np.array(history_3)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 3.0)
plt.savefig(f"./OutputGraph/{model3_name}/loss.png")
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(f"./OutputGraph/{model3_name}/acc.png")
plt.show()

history = np.array(history_4)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 2.0)
plt.savefig(f"./OutputGraph/{model4_name}/loss.png")
plt.show()


history = np.array(history_ensemble)
plt.plot(history[:, 0:2])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(f"./OutputGraph/{model5_name}/acc.png")
plt.show()
print('--------------------结果可视化完成~~-----------------------------------------------------')
print('--------------------训练完成~~-----------------------------------------------------')