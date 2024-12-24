# 检测images和labels中的文件是否一一对应

import os

# 修改输入图片文件夹
img_folder = r"D:\PyCharmWorkSpace\spd\spd\pc_data\images"
img_list = os.listdir(img_folder)

# 修改输入标签文件夹
label_folder = r"D:\PyCharmWorkSpace\spd\spd\pc_data\labels"
label_list = os.listdir(label_folder)


for img in img_list:
    if img.replace(".jpg", ".txt") not in label_list:
        print(f"Label file not found for image: {img}")
