# 动态权重调整与模型融合项目

本仓库包含了动态权重调整系统用于模型融合的实现，使用了BP神经网络和DenseNet121模型。同时，还包括了基于投票方法的单个模型和集成模型的代码。

## 1 项目概述

​     本项目旨在动态调整两种不同模型——BP和DenseNet121的权重，以在图像分类任务中获得更好的性能。系统采用数据驱动的方法动态调整参数，融合模型以提高准确率。

![融合模型结构](.\our_model.jpg)

## 2 使用方法

### 2.1环境设置

设置环境时，请克隆仓库并安装所需的依赖项：

```bash
git clone https://github.com/yourusername/dynamic-weight-adjustment-fusionmodel.git
cd dynamic-weight-adjustment-fusionmodel
pip install -r requirements.txt
```

### 2.2数据

- `source.xlsx`: 文件包含了用于训练的多光谱数据和可见光图片的地址。注：地址随意不能进行更换

- `images` :文件夹包含了用于训练的图片。

- `Data_Augmentation.py` :数据增强代码，具体增强方式包括随机镜像翻转和随机切割，通过调整文件中的flag参数，可以指定每张图片的增强次数。

### 2.3模型

- `our_model.py`：融合模型的实现。
- `单个模型文件`：每个文件以其代表的模型命名。
- `Ensembel_集成模型文件.py`：每个文件以其模型名称以`Ensemble`命名，表示它使用投票方法进行模型融合,因为是对结果进行融合，所有的单个模型是单独进行训练，也可以对每个单独的模型进行训练。

### 2.4结果

- `record` 文件夹包含了模型训练过程中产生的CSV文件，文件记录了模型训练时产生的Loss以及Accuracy等情况。

- `Outputimages` 文件夹包含了模型训练结束后的训练情况曲线图。

## 3 贡献

欢迎对项目进行贡献。请fork仓库并提交pull request以改进项目。

## 4 许可证

