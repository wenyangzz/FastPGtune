import time

i = 1  # 示例值，根据实际情况定义i的值
time_1 = time.time()
print("第轮结束的系统时间是{}\n".format(time.ctime()))
time.sleep(10)
time_2 = time.time()
print("第轮结束的系统时间是{}\n".format(time.ctime()))
print("这一轮耗时: {:.4f} seconds\n".format(time_2 - time_1))