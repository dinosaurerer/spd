# Intelligent Sitting Posture Detection System (MY-PDS)
 Deep learning sitting posture detection (SPD) based on multimodal datasets

**基于Mediapipe骨骼提取与YOLOV11-MSCA改进模型的人体坐姿检测系统 (MY-PDS)**

***Sitting posture detection system based on Mediapipe bone extraction and YOLOV11-MSCA model***

## 项目概述

本项目旨在通过深度学习技术搭建一个智能坐姿检测系统。利用Mediapipe和改进后的YOLOV11-MSCA模型，系统能够实时监测用户坐姿并提供及时反馈。项目将通过骨骼架构提取和多尺度通道注意力机制提升坐姿检测的精度与鲁棒性，结合多模态数据集，确保系统在多种环境下高效运行。同时，还计划加入语音提示、个性化健康建议等创新功能，以提升用户的交互体验和健康意识。

## 项目目标
- **用户坐姿多分类**：用户坐姿多分类：支持 6 类坐姿检测与分类，具体类别如下：
- ![demo](pngs/classes.png)
- **实时坐姿检测**：基于Mediapipe骨骼提取和YOLOv11-MSCA模型进行实时坐姿检测。
- **精准错误坐姿识别**：高效识别常见的错误坐姿，如前倾、侧倾等，并提供实时反馈。
- **多模态数据支持**：融合不同场景（教室、车内、办公室等）的多模态数据集，提升模型的泛化能力。
- **健康情况统计**：根据视频或图片生成坐姿分析结果表，后续可生成用户专属健康报告与改进建议。

## 功能亮点

1. **创新的人体骨骼提取与检测流程**：结合Mediapipe骨骼提取与YOLOV11-MSCA模型，去除背景与干扰，提升模型鲁棒性，尤其在复杂环境中。
2. **改进的YOLOv11-MSCA模型**：在YOLOv11基础上，集成了多尺度通道注意力（MSCA）机制，优化检测精度与计算效率。
3. **多模态数据集支持**：支持多场景数据集（教室、车内、办公室等），提高模型的适用性与稳定性。
4. **实时性与高效性**：优化后的YOLOV11-MSCA模型，具备高效推理速度，适合边缘设备与嵌入式平台实时部署。
5. **多种输入源支持**：支持从图片、视频、摄像头、文件夹（批量）和网络摄像头等多种输入源进行目标检测，满足不同场景下的使用需求。
6. **精美的GUI设计**：直观友好的图形用户界面，提供流畅的交互体验与自定义功能。

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
```

### 克隆本项目到本地：
```bash
git clone https://github.com/dinosaurerer/spd.git
cd spd/GUI
```

### 启动坐姿检测系统

根据你所使用的操作系统，选择合适的启动方式：

#### **1. Windows系统**
在 Windows 上，你可以直接双击 `run.bat` 文件来启动坐姿检测系统，或者在命令行中执行以下命令：

#### **2. Linux / macOS系统**   
在 Linux 或 macOS 上，你需要使用 run.sh 脚本来启动坐姿检测系统。首先，确保你有执行权限：
```batch
chmod +x run.sh
```
然后运行 run.sh 脚本：
```batch
./run.sh
```

#### **3.使用 Python 脚本直接启动**
如果你不想使用脚本，可以直接通过命令行运行 Python 脚本：
```batch
python main.py
```


## 项目截图
![demo](pngs/gui.jpg)


## 项目结构
```bash
spd/
├── GUI/                         # 图形用户界面模块
│   ├── gui.py                   # 启动脚本，坐姿检测系统的主入口
│   ├── run.sh                   # Linux/macOS 启动脚本
│   ├── run.bat                  # Windows 启动脚本
│   └── ...                      # 其他GUI相关文件
├── YOLO/                        # YOLOV11-MSCA模型相关文件
│   ├── train.py                 # YOLO模型训练脚本
│   ├── config.yaml              # 模型配置文件
│   ├── datasetst/MSD            # 数据集
│   ├── ultralytics/             # 更改MSCA模型结构
│   └── ...                      # 其他YOLO相关内容
├── demo_video/                  # 演示视频存放目录
├── pngs/                        # 模型效果展示图片存放目录
│   ├── result1.png
│   ├── result2.png
│   └── ...
├── process_data/                # 数据处理相关文件
├── LICENSE                      # 项目许可证
├── README.md                    # 项目说明文档
└── requirements.txt             # Python依赖项清单

```

## 模型下载
- MediaPipe 模型：[下载链接](https://raw.gitcode.com/open-source-toolkit/ef632/blobs/764b9cffcbd182c8acb21d734c4b2a0c7309adbe/mediapipe_tf_file.rar)
- YOLOv11 模型：[下载链接](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes)
- YOLOv11-MSCA模型：[下载链接](https://storage.googleapis.com/kaggle-models-data/177835/208593/bundle/archive.tar.gz?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1735291266&Signature=Zg2as1d6x%2BK0O1gO5Z4g9AxduJeFFNEYP1nwVzst3HrhEnr7U8HOnrBqrKA%2BbzBQngZyAJSa1iSoysaf11hR48F8QRSIOyYVy5IjmAbYg3tErujaAY8oblwZCys9HMZ5A0A9vNLqV5Eel8BRRmJVQ8J15y1g10srSzfhFjtmJd7gBV2SRynsARRGJus7dq7tO9WsZALG3fF%2BH%2BvzOfOWFMwwsmLPYkiG25S9h0Y6UOi00DBB28jQuLuE8JfK%2ByVOldcoP93F%2FxccEpAkUY5KIIMd0FYTJo9H9vYzrQnC%2BJL%2B6cDHbpltvFdmYFfAK05TmECiRgs2b0ObCojOEjzSAQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dyolov11-msca-pytorch-default-v1.tar.gz)
- 改进模型ultralytics库文件：[下载链接](https://storage.googleapis.com/kaggle-data-sets/6365474/10286135/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241224T092456Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1f00315bd368b2521a0384507ef0bf51822f45776ed51be32a1467bb6c463f3aa98ea6f8b3231bd5b483a991733f52bfdd9a71c58be0a3a7e0046456294096868b87193a9b071d5a1d214d1fb5f5535cfafc9fb8eacb124a96288581363987bffe0750b74580d72ceb72483d2a17fa2a940c1cee553a95e997dc0313cad678fcc59be4d526beb6992931757f1489a2c0ab973145603ceb2b886313f5facf081af9e8669b41ad61ef355339dc068cfa5db5f87cf6e9b1cc610d2618d9c44d1f5f8c8b264cdee1ac5af4e9436d3a27693122fd5d0630efbd9974f5494001fa9590ecb6565b776f4653239258295ade930279a94ba84c82b224906fd3f69c605406)
## 模型效果
- 训练结果
- ![demo](pngs/res.png)
- 训练损失曲线
- ![demo](pngs/loss.png)
- F1曲线
- ![demo](pngs/F1-Curve.png)
- PR曲线
- ![demo](pngs/PR-Curve.png)

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
     
    - YOLOV11-MSCA模型的kagglehub 下载：
   ```bash
     import kagglehub

     # Download latest version
     path = kagglehub.model_download("xiaokonglong80/yolov11-msca/pyTorch/default")
    
     print("Path to model files:", path)