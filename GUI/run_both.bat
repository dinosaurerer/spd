#!/bin/bash


# 运行 main.py 打开主程序
python main.py &

# 运行 mp4.py 播放视频
python mp4.py &

# 等待所有后台进程完成
wait
