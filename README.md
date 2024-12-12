# sitting posture detection
 Deep learning sitting posture (SPD) detection based on multimodal datasets

# 基于深度学习的智能坐姿检测与改进

## 项目概述
该项目旨在通过深度学习技术，搭建一个智能坐姿检测系统。通过实时分析用户的坐姿，系统能够识别坐姿是否正确并提供即时反馈。项目使用了 MediaPipe 与 YOLOv8 模型进行姿势识别，并通过语言提示来提醒用户改正错误坐姿。为了增强体验，本项目还计划加入多模态数据融合、增强现实（AR）反馈、个性化健康建议等创新功能。

## 项目目标
- **实时坐姿检测**：使用 MediaPipe 和 YOLOv8 对用户坐姿进行实时检测。
- **错误坐姿识别**：将坐姿分为多个分类，识别出不正确的坐姿并给予语言提示。
- **创新功能**：
  - 融合语音识别和动作检测，提供更多维度的反馈。
  - 引入增强现实技术，通过虚拟助手或3D模型为用户提供实时坐姿纠正指导。
  - 个性化健康建议，生成用户专属的坐姿改进报告。

## 功能亮点
1. **实时坐姿检测**：使用YOLOv8，实现对用户坐姿的高效实时监控与分析。
2. **个性化健康建议**：基于坐姿数据分析，提供个性化的姿势改进建议，并在系统中加入久坐提醒功能。
3. **多模态交互体验**：结合语音和手势识别，为用户提供更加灵活和多维度的交互体验。
4. **增强现实反馈**：通过AR技术实时展示虚拟反馈，帮助用户直观地理解正确与错误坐姿。
5. **轻量化模型支持低功耗设备**：通过模型剪枝、量化技术，优化系统在低功耗设备上的运行效率，支持智能手机和嵌入式设备。

## 安装与使用

### 环境要求
- Python>=3.8
- PyTorch
- OpenCV
- YOLOv11
- TensorFlow (若使用AR功能)
- 其他依赖：可以通过以下命令安装依赖项：

```bash
pip install -r requirements.txt

克隆本项目到本地：

git clone https://github.com/your-username/pose-detection.git
cd pose-detection

启动坐姿检测系统：
python main.py
```


## 项目截图
![demo](pngs/d1.jpg)

## 示例输出
当系统检测到错误的坐姿时，屏幕上将显示如下警告
![ww](pngs/d1.jpg)

## 项目结构
```bash
pose-detection/
├── data/                  # 数据集
│   ├── train/             # 训练集
│   ├── test/              # 测试集
├── models/                # 训练好的深度学习模型
├── src/                   # 源代码
│   ├── main.py            # 主程序
│   ├── detector.py        # 坐姿检测模块
│   ├── ar_feedback.py     # AR反馈模块
│   └── health_advice.py   # 健康建议模块
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 模型下载
- MediaPipe 模型：[下载链接](https://raw.gitcode.com/open-source-toolkit/ef632/blobs/764b9cffcbd182c8acb21d734c4b2a0c7309adbe/mediapipe_tf_file.rar)
- YOLOv8 模型：[下载链接](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes)
- v8剪枝蒸馏模型：[下载链接](https://drive.google.com/...)

## 模型效果
- 混淆矩阵
![demo](pngs/d2.png)
- 训练损失曲线
![demo](pngs/d3.png)

## 贡献
欢迎大家为项目做出贡献！如果你有好的功能建议、bug修复或优化方案，请提起一个 Pull Request 或 Issue。

## 许可协议
本项目采用 [MIT License](LICENSE)。

## 致谢
感谢开源社区提供的 MediaPipe 和 YOLOv8 模型，它们为本项目的实施提供了很大的帮助。
感谢团队合作者[杨刘军](https://github.com/Future-Elite)和[邹杰](https://github.com/Ezequiel041223)对项目的巨大贡献，也感谢所有关注和支持本项目的用户！
本项目灵感来自于人工智能领域的坐姿矫正技术研究，旨在提升用户的健康坐姿意识。
更多细节将会随着项目的迭代更新，感谢您的关注与支持！

### 更新说明：
1. 数据集部分被详细列出，并展示了数据集结构。
   - 数据集当前已在kaggle开源：
     - [kaggle直链下载](https://storage.googleapis.com/kaggle-data-sets/6227602/10097754/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241204%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241204T101624Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=42fae711ffd7af00659a7e45b273c9cff463e6c14924290d2d5cb144d3f5caa6eebd557c160a798526464f5f73c226133bd831a80c962a9f0a0fbe780ad2e0060e6feb0bdacf74c97b244681d771d73866d0db4bcf5ee9fccf2f4409fa772369cbd1f8ad606b138fc9983c5c40ef6520482330c1be2f28d999ecfa451d9a3f1552ca49a5d7bf45d07b4d6051c7b4bd1378f1acbf77307e1caaf2e7d909911d5db74381c40710c013ac1e54071f23a6a8ef3b10d36b7e3063010741e4a26c74a1b6d4a1ddf6fdc7f72926a365603acb32a9335ce2b64ee4ce76b63cfcbfca4385004a44af7300c7c3c67e4e043331c74896957fbfc67540022980927c58965b8a)
     - kagglehub 下载
     ```bash
     import kagglehub

     # Download latest version
     path = kagglehub.dataset_download("xiaokonglong80/spd-yolo")
  
     print("Path to dataset files:", path)
