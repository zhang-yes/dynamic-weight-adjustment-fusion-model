'''
    读取excel表格,将图片进行数据增强
    并将图片的地址保存到一个excel中
    增强后的图片与原始图片在同一个文件夹下
'''
flag = 12  # 增强次数
src_path = './source.xlsx'  # 待增强的excel表单数据
out_path = './hence.xlsx'  # 增强过后的excel表单数据

import pandas as pd
from PIL import Image
import os
import random
from tqdm import tqdm

# 读取Excel文件
df = pd.read_excel(src_path)


# 定义图片增强函数
def random_crop(image, target_size):
    width, height = image.size
    if width < target_size[0] or height < target_size[1]:
        # 如果图片小于目标尺寸，首先调整图片大小
        image = image.resize((max(width, target_size[0]), max(height, target_size[1])))
        width, height = image.size
    # 确保裁剪范围有效
    left = random.randint(0, width - target_size[0])
    top = random.randint(0, height - target_size[1])
    return image.crop((left, top, left + target_size[0], top + target_size[1]))


def random_flip(image):
    if random.random() < 0.25:
        return image.transpose(Image.FLIP_LEFT_RIGHT)  # 随机水平翻转
    if random.random() < 0.5:
        return image.transpose(Image.FLIP_TOP_BOTTOM)  # 随机水平翻转
    return image  # 不进行翻转


# 目标尺寸
target_size = (1350, 4350)
# 遍历Excel中的图片地址列
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    image_path = row['图片地址']
    image = Image.open(image_path)
    if image is None:
        print(f"无法打开图片：{image_path}")
        continue
    # 原始图片信息保持不变
    new_row_data = row.to_dict()
    # 数据增强：随机切割、随机旋转和随机镜像翻转
    for i in range(flag):
        new_image = image.copy()  # 复制原始图片以避免修改原图
        new_image = random_crop(new_image, target_size)  # 随机切割
        if new_image is None:
            print(f"无法裁剪图片到目标尺寸：{image_path}")
            continue
        new_image = random_flip(new_image)  # 随机镜像翻转
        # 保存新图片
        new_image_name = f"new{i + 1}_{os.path.basename(image_path)}"
        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        new_image.save(new_image_path)
        # 更新Excel文件
        new_row_data['图片地址'] = new_image_path
        df = df._append(new_row_data, ignore_index=True)
# 保存更新后的Excel文件
df.to_excel(out_path, index=False, )