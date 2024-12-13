import sys
import os
import logging
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QProgressBar

# 将ui目录添加到系统路径中
sys.path.append(os.path.join(os.getcwd(), "ui"))
# 禁止标准输出
sys.stdout = open(os.devnull, 'w')
logging.disable(logging.CRITICAL)  # 禁用所有级别的日志
from utils import glo



class SplashScreen(QLabel):
    """启动动画窗口"""
    def __init__(self, parent=None):
        super(SplashScreen, self).__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框，置顶显示
        self.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.setPixmap(QPixmap("images/splash_logo.png"))  # 设置Logo图片
        self.setStyleSheet("background-color: white;")  # 设置背景色
        self.resize(1140, 900)  # 设置窗口大小



if __name__ == '__main__':
    app = QApplication([])  # 创建应用程序实例
    app.setWindowIcon(QIcon('images/yoloshow.ico'))  # 设置应用程序图标

    # 为整个应用程序设置样式表，去除所有QFrame的边框
    app.setStyleSheet("QFrame { border: none; }")

    # 创建启动动画窗口
    splash = SplashScreen()
    splash.show()

    # 创建主窗口实例（延迟加载）
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
    QTimer.singleShot(1, load_main_window)

    app.exec()  # 启动应用程序的事件循环
