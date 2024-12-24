import os
import shutil


def get_next_index(folder_path):
    """
    获取文件夹中最大的索引值（以文件名的序号为准），返回下一个可用的索引值。
    :param folder_path: 文件夹路径
    :return: 下一个可用的索引值
    """
    existing_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    indices = []
    for file_name in existing_files:
        try:
            index = int(file_name.split('_')[-1].split('.')[0])  # 提取文件名中的索引部分
            indices.append(index)
        except (ValueError, IndexError):
            continue
    return max(indices, default=0) + 1


def copy_and_rename_images(src_base, dest_base):
    """
    复制并重命名图片，将用户1的数据集合并到目标文件夹，并确保文件名序号连续。
    :param src_base: 源文件夹路径
    :param dest_base: 目标文件夹路径
    """
    for class_folder in os.listdir(src_base):
        src_folder = os.path.join(src_base, class_folder)
        dest_folder = os.path.join(dest_base, class_folder)

        # 检查是否为有效文件夹
        if not os.path.isdir(src_folder) or not os.path.isdir(dest_folder):
            print(f"Skipping {class_folder}: Not a valid folder.")
            continue

        # 获取目标文件夹的下一个起始索引
        next_index = get_next_index(dest_folder)

        # 遍历源文件夹中的图片
        for file_name in os.listdir(src_folder):
            if file_name.endswith('.jpg'):
                src_file = os.path.join(src_folder, file_name)

                new_file_name = f"{class_folder[2:]}_{next_index:03d}.jpg"
                dest_file = os.path.join(dest_folder, new_file_name)

                # 复制并重命名文件
                shutil.copy(src_file, dest_file)
                print(f"Copied and renamed {src_file} to {dest_file}")

                next_index += 1


if __name__ == "__main__":
    src_base = r"C:\Users\lenovo\Desktop\y"  # 用户1数据集路径
    dest_base = r"D:\\PyCharmWorkSpace\\spd\\spd\\pc_data"  # 目标数据集路径

    # 确保源和目标路径都存在
    if not os.path.exists(src_base):
        print(f"Source path does not exist: {src_base}")
    elif not os.path.exists(dest_base):
        print(f"Destination path does not exist: {dest_base}")
    else:
        copy_and_rename_images(src_base, dest_base)
        print("All files copied and renamed successfully!")