#!/bin/bash


# 运行 main.py 打开主程序
python gui.py &

# 运行 splash.py 播放视频
python splash.py &

# 等待所有后台进程完成
wait
