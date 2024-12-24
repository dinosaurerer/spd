import os

# 类名索引对应字典
CLASS_MAPPING = {
    "normal": 0,
    "head_tilt_left": 1,
    "head_tilt_right": 2,
    "body_left": 3,
    "body_right": 4,
    "left_support_head": 5,
    "right_support_head": 6,
    "lying_down": 7,
    "head_forward": 8
}

def get_class_index(file_name):
    """
    根据文件名提取类名索引
    :param file_name: txt文件名
    :return: 类名索引
    """
    for class_name, index in CLASS_MAPPING.items():
        if class_name in file_name:
            return index
    raise ValueError(f"Unknown class in file name: {file_name}")

def filter_boxes(lines):
    """
    根据检测框面积过滤掉较小的框
    :param lines: txt文件中的所有行
    :return: 保留的行
    """
    if not lines:
        return lines

    # 计算面积并排序
    boxes = [(line, float(line.split()[3]) * float(line.split()[4])) for line in lines]
    boxes.sort(key=lambda x: x[1], reverse=True)

    # 仅保留面积最大的框
    return [boxes[0][0]]

def process_labels(labels_folder):
    """
    遍历labels文件夹，更新类名索引并过滤目标框
    :param labels_folder: labels文件夹路径
    """
    for file_name in os.listdir(labels_folder):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(labels_folder, file_name)

        # 获取文件对应的类名索引
        class_index = get_class_index(file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # 更新类名索引并过滤框
        updated_lines = [
            f"{class_index} {line.split(maxsplit=1)[1]}"
            for line in filter_boxes(lines)
        ]

        # 写回文件
        with open(file_path, "w") as f:
            f.writelines(updated_lines)

if __name__ == "__main__":
    labels_folder = r"D:\\PyCharmWorkSpace\\spd\\spd\\pc_data\\labels"

    if not os.path.exists(labels_folder):
        print(f"Labels folder does not exist: {labels_folder}")
    else:
        process_labels(labels_folder)
        print("Labels updated successfully!")
