import sys
import os
import logging
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from utils import glo


# 将ui目录添加到系统路径中
sys.path.append(os.path.join(os.getcwd(), "ui"))

# 禁止标准输出
logging.disable(logging.CRITICAL)  # 禁用所有级别的日志


class VideoPlayerThread(QThread):
    """视频播放线程，避免阻塞主线程"""
    frame_update = Signal(QPixmap)

    def __init__(self, video_path):
        super(VideoPlayerThread, self).__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频播放完毕
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.frame_update.emit(pixmap)  # 发射信号，更新视频帧
            QThread.msleep(int(1000 / self.fps))  # 根据帧率暂停

        self.cap.release()  # 释放视频捕捉资源


class SplashScreen(QWidget):
    """启动动画窗口播放视频"""
    def __init__(self, parent=None):
        super(SplashScreen, self).__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框，置顶显示

        # 创建标签用来显示视频帧
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # 居中显示

        # 加载视频并启动视频播放线程
        self.video_path = 'images/splash.mp4'
        self.video_thread = VideoPlayerThread(self.video_path)
        self.video_thread.frame_update.connect(self.update_frame)  # 连接信号到槽
        self.video_thread.start()

        # 获取视频的尺寸，设置窗口大小
        self.resize(self.video_thread.frame_width, self.video_thread.frame_height)

        # 设置窗口的布局为居中显示
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_frame(self, pixmap):
        """更新视频帧"""
        self.label.setPixmap(pixmap)
        self.label.setFixedSize(pixmap.size())

    def closeEvent(self, event):
        """当窗口关闭时，停止视频播放线程"""
        self.video_thread.quit()
        self.video_thread.wait()
        event.accept()


def load_main_window():
    """加载主窗口"""
    from yoloshow.Window import YOLOSHOWWindow as yoloshowWindow
    from yoloshow.Window import YOLOSHOWVSWindow as yoloshowVSWindow
    from yoloshow.ChangeWindow import yoloshow2vs, vs2yoloshow

    # 创建主窗口和对比窗口
    yoloshow = yoloshowWindow()
    yoloshowvs = yoloshowVSWindow()

    # 初始化全局变量管理器，并设置值
    glo._init()  # 初始化全局变量空间
    glo.set_value('yoloshow', yoloshow)  # 存储yoloshow窗口实例
    glo.set_value('yoloshowvs', yoloshowvs)  # 存储yoloshowvs窗口实例

    # 从全局变量管理器中获取窗口实例
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')

    # 显示yoloshow窗口
    yoloshow_glo.show()

    # 连接信号和槽，以实现界面之间的切换
    yoloshow_glo.ui.src_vsmode.clicked.connect(yoloshow2vs)  # 从单模式切换到对比模式
    yoloshowvs_glo.ui.src_singlemode.clicked.connect(vs2yoloshow)  # 从对比模式切换回单模式

    splash.close()  # 关闭启动动画窗口


if __name__ == '__main__':
    app = QApplication([])  # 创建应用程序实例
    app.setWindowIcon(QIcon('images/icon.ico'))  # 设置应用程序图标

    # 为整个应用程序设置样式表，去除所有QFrame的边框
    app.setStyleSheet("QFrame { border: none; }")

    # 创建启动动画窗口并显示
    splash = SplashScreen()
    splash.show()

    # 延迟执行主窗口加载
    QTimer.singleShot(3000, load_main_window)  # 延迟3秒后加载主窗口

    app.exec()  # 启动应用程序的事件循环
