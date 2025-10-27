import sys
import tracemalloc
import time
import os

sys.path.append("..") 

from optimizer_pobo_sa2 import PollingBayesianOptimization
from utils2 import RealEnv

if __name__ == '__main__':
    # 开始跟踪内存分配
    tracemalloc.start()

    # prepare the environment
    env = RealEnv()
    model = PollingBayesianOptimization(env, seed=1, threshold=0.95)
    
    # initial sampling
    model.init_sample()
    log_dir = "/home/yinzh/VDTuner/diskann/msong/q_10"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "Vamana_10_msong.log")

    # 以 'w' 模式打开文件一次，初始化文件内容
    with open(log_file_path, 'a') as f:
        f.write("开始时的系统时间{}\n".format(time.ctime()))
        
    total_time = 0  # 初始化总时间为0
    
    # iterative auto-tuning
    start_time = time.time()  # 开始计时
    for i in range(10):
        # 以 'a' 模式追加写入
        with open(log_file_path, 'a') as f:
            f.write("第{}轮开始的系统时间是{}\n".format(i, time.ctime()))
        time_1 = time.time()
        model.step1(10)
        time_2 = time.time()
        with open(log_file_path, 'a') as f:
            f.write("第{}轮结束的系统时间是{}\n".format(i, time.ctime()))
            f.write("这一轮耗时: {:.4f} seconds\n".format(float(time_2 - time_1)))
        current, peak = tracemalloc.get_traced_memory()
        with open(log_file_path, 'a') as f:
            f.write(f"Current memory usage: {current / 1024 / 1024:.2f} MB\n")
    end_time = time.time()
    total_time = end_time - start_time  # 计算总时间
    # 获取当前内存和峰值内存
    current, peak = tracemalloc.get_traced_memory()
    # 将信息写入文件
    with open(log_file_path, 'a') as f:
        f.write("结束时的系统时间{}\n".format(time.ctime()))
        f.write(f"Current memory usage: {current / 1024 / 1024:.2f} MB\n")
        f.write(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB\n")
        f.write(f"Total time for all iterations: {total_time:.4f} seconds\n")

    # 停止跟踪
    tracemalloc.stop()