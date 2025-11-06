import re
import csv

# 读取文件内容
file_path = 'record.log'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 定义正则表达式模式
pattern = r'\[(\d+)\]\s+(\d+)\s+\{([^}]+)\}\s+\{([^}]+)\}\s+([\d.]+)\s+([\d.]+)\s+(\d+)'

# 解析数据
data = []
for line in lines:
    match = re.match(pattern, line)
    if match:
        row_data = [
            int(match.group(1)),
            int(match.group(2)),
            eval(f"{{{match.group(3)}}}"),
            eval(f"{{{match.group(4)}}}"),
            float(match.group(5)),
            float(match.group(6)),
            int(match.group(7))
        ]
        data.append(row_data)

# 提取y1和y2
recall_values = [row[-3] for row in data]
time_values = [row[-2] for row in data]

# 设置阈值
thresholds = [0.9, 0.925, 0.95, 0.975]

# 初始化当前最小的y2值
current_min_time = [300, 300, 300, 300]

min_time_values_9 = []
min_time_values_925 = []
min_time_values_95 = []
min_time_values_975 = []

for recall, time in zip(recall_values, time_values):
    if recall > thresholds[0] and time < current_min_time[0]:
        current_min_time[0] = time
    min_time_values_9.append(current_min_time[0])

    if recall > thresholds[1] and time < current_min_time[1]:
        current_min_time[1] = time
    min_time_values_925.append(current_min_time[1])

    if recall > thresholds[2] and time < current_min_time[2]:
        current_min_time[2] = time
    min_time_values_95.append(current_min_time[2])

    if recall > thresholds[3] and time < current_min_time[3]:
        current_min_time[3] = time
    min_time_values_975.append(current_min_time[3])

# 将y1、y2和当前最小的y2存储到CSV文件中
csv_file_path = 'y1_y2.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Row Number', 'recall', 'time','Min time when y1 > 0.9', 'Min time when y1 > 0.925', 'Min time when y1 > 0.95', 'Min time when y1 > 0.975'])
    for i, (recall, time,min_time_9, min_time_925, min_time_95, min_time_975) in enumerate(zip(
        recall_values, time_values, min_time_values_9, min_time_values_925, min_time_values_95, min_time_values_975
    )):
        writer.writerow([i + 1, recall, time, time,min_time_9, min_time_925, min_time_95, min_time_975])