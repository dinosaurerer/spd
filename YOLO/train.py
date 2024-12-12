import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-MSCAAttention1.yaml')  # 地址改成自己的
    model.train(data=r'coco128.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )
