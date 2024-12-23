import logging
import os
import sys

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from utils import glo

# 将ui目录添加到系统路径中
sys.path.append(os.path.join(os.getcwd(), "ui"))

# 禁止标准输出
logging.disable(logging.CRITICAL)  # 禁用所有级别的日志


class SplashScreen(QWidget):
    """启动动画窗口播放视频"""
    def __init__(self, parent=None):
        super(SplashScreen, self).__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框，置顶显示

        # 创建标签用来显示视频帧
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # 居中显示
        self.label.setContentsMargins(0, 0, 0, 0)  # 去除QLabel的内边距

        # 加载视频
        self.video_path = './images/splash.mp4'
        self.cap = cv2.VideoCapture(self.video_path)

        # 获取视频的帧率和尺寸
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置缩放比例（0.5表示缩小为原来的一半）
        self.scale_factor = 0.5
        self.frame_width = int(self.frame_width * self.scale_factor)
        self.frame_height = int(self.frame_height * self.scale_factor)

        # 设置窗口大小与缩放后的尺寸一致
        self.resize(self.frame_width, self.frame_height)

        # 设置窗口的布局为居中显示
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)  # 去除布局的内边距
        self.setLayout(layout)

        # 定时器，用来每隔一段时间更新视频帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(2500 / self.fps))  # 根据视频帧率更新

        # 居中显示窗口
        self.center_window()

    def center_window(self):
        """将窗口居中显示"""
        screen_geometry = QGuiApplication.primaryScreen().geometry()
        screen_center = screen_geometry.center()  # 获取屏幕中心
        window_geometry = self.geometry()
        window_center = window_geometry.center()  # 获取窗口当前尺寸
        # 将窗口的中心设置为屏幕的中心
        self.move(screen_center - window_center)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 将BGR格式的OpenCV图像转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 按照缩放比例缩小图像
            frame_rgb = cv2.resize(frame_rgb, (self.frame_width, self.frame_height))

            # 转换为QImage格式
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # 更新标签显示的图片
            self.label.setPixmap(pixmap)

            # 调整label大小，确保视频填满整个窗口
            self.label.setFixedSize(pixmap.size())

        else:
            # 如果视频播放完毕，关闭启动窗口
            self.cap.release()
            self.close()


if __name__ == '__main__':
    app = QApplication([])  # 创建应用程序实例
    app.setWindowIcon(QIcon('images/icon.ico'))  # 设置应用程序图标

    # 为整个应用程序设置样式表，去除所有QFrame的边框
    app.setStyleSheet("QFrame { border: none; }")

    # 创建启动动画窗口并显示
    splash = SplashScreen()
    splash.show()

    # 创建主窗口实例
    def load_main_window():
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

    # 延迟执行主窗口加载
    # QTimer.singleShot(3000, load_main_window)  # 延迟3秒后加载主窗口

    app.exec()  # 启动应用程序的事件循环
