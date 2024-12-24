import subprocess
import os

# 获取虚拟环境中的 Python 解释器路径

python_path = os.path.join(r'D:\PyCharmWorkSpace\DeepLearning\venv', 'Scripts', 'python.exe')  # Windows

# 运行 splash.py 使用虚拟环境的 Python 解释器
process1 = subprocess.Popen([python_path, 'splash.py'])

# 运行 gui.py 使用虚拟环境的 Python 解释器
process2 = subprocess.Popen([python_path, 'gui.py'])

# 等待两个进程结束
process1.wait()
process2.wait()
