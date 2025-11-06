import csv
import sys
import os

sys.path.append("..")
import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as  sp

import traceback

KNOB_PATH = r'/home/yinzh/VDTuner/auto-configure/whole_param_hnsw.json'
RUN_ENGINE_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/run_engine.sh'

LOG_DIR = r'/home/yinzh/VDTuner/hnsw/msong/q_10/log/'
PROJECT_DIR = r'/home/yinzh/VDTuner/'

DATA_PATH = r"/mnt/data2/zhou/dataset/enron/enron_base.fbin"
GTRUE = r"/mnt/data2/zhou/dataset/enron/enron_groundtruth.ivecs"
QUERY_PATH = r"/mnt/data2/zhou/dataset/enron/enron_query.fvecs"

DATA_SET = r"msong"

def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)

    return latin_samples

class KnobStand:
    def __init__(self, path) -> None:
        self.path = path
        with open(path, 'r') as f:
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name, zero_one_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'float':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return float(real_val), float(real_val)   

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'float':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class StaticEnv:
    def __init__(self, model_path=['XGBoost_20knob_thro.model', 'XGBoost_20knob_prec.model'], knob_path=r'milvus_important_params.json') -> None:
        self.model_path = model_path
        self.get_surrogate(model_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_surrogate(self, surrogate_path):
        # surrogate1, surrogate2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])
        self.model1, self.model2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])

    def get_state(self, knob_vals_arr):
        Y1, Y2 = [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"Index parameters changed: {conf_value}")

            y1 = self.model1.predict([conf_value])[0]
            y2 = self.model2.predict([conf_value])[0]

            self.sampled_times += 1
            print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1} {y2}')
            
            Y1.append(y1)
            Y2.append(y2)
        return np.concatenate((np.array(Y1).reshape(-1,1), np.array(Y2).reshape(-1,1)), axis=1)

class RealEnv:
    def __init__(self, bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH) -> None:
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_state(self, knob_vals_arr):
        # NSG
        def read_file(_filename):
            performance_file = _filename + "_performance.csv"
            if not os.path.exists(performance_file):
                return min(self.Y1_record), min(self.Y2_record)
            with open(performance_file, 'r') as file:
                reader = csv.reader(file)
                # 只读取第一行
                first_row = next(reader)
                # 将所有值转换为浮点数
                performance = [float(value) for value in first_row[1:]]
                return performance[0], performance[1]
        # KGraph
        # def read_file(filename):
        #     with open(filename) as f:
        #         content = f.read()
        #     pattern = r'recall:\s*([\d\.e\+]+).*time:\s*([\d\.e\+]+)'
        #     log_matches = re.findall(pattern, content)
        #     if log_matches:
        #         match = log_matches[-1]
        #         return float(match[0]), float(match[1])
        #     return min(self.Y1_record), max(self.Y2_record)
        Y1, Y2, Y3 = [], [], []
        print(knob_vals_arr)
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(record)]
            # change
            index_value, system_value = conf_value[:4], conf_value[4:]
            index_name, system_name = self.names[:4], self.names[4:]

            index_conf = dict(zip(index_name,index_value))
            system_conf = dict(zip(system_name,system_value))
            try:
                # Y1 为recall，Y2为构图时间
                if index_conf["index_type"] == 'KGraph':
                    if index_conf["S"] > index_conf["L"]:
                        index_conf["S"] = index_conf["L"]
                    filename = "_".join([f"{key}_{value}" for key, value in index_conf.items() if key != "index_type"])
                    log_filename = LOG_DIR + filename

                    cmd_args = " ".join([f"-{key} {value}" for key, value in index_conf.items() if key!= "index_type"])
                    if not os.path.exists(log_filename):
                        print("bash %sscripts/run_kgraph_index.sh --output %s --log-file %s %s " % (PROJECT_DIR,
                                                                                                    'kgraph_index',
                                                                                                    log_filename,
                                                                                                    cmd_args))
                        os.system("bash %sscripts/run_kgraph_index.sh --output %s --log-file %s %s " % (PROJECT_DIR,
                                                                                                    'kgraph_index',
                                                                                                    log_filename,
                                                                                                    cmd_args))
                    y1, y2 = read_file(log_filename)
                elif index_conf["index_type"] == 'hnsw':
                    # if index_conf["L"] < index_conf["R"]:
                    #     index_conf["L"] = index_conf["R"]
                    filename ="_".join([f"{k}{v}" for k, v in sorted(index_conf.items()) if k != "index_type"])
                    filename = f"{LOG_DIR}{filename}"
                    cmd = ["1"]
                    # diskann_dir = str(PROJECT_DIR + "diskann/" + "graph/" + "diskann_" + DATA_SET)
                    cmd_tmp = " ".join([f"{value}" for value in index_conf.values() if value != index_conf["index_type"]])
                    cmd.append(cmd_tmp)
                    # cmd.append(diskann_dir)
                    # cmd.append(QUERY_PATH)
                    # cmd.append(GTRUE)
                    cmd.append(filename)
                    cmd_args = " ".join(cmd)
                    performance_file = filename + "_performance.csv"
                    # if not os.path.exists(performance_file):
                    print("%sscripts/test_hnsw %s" % (PROJECT_DIR, cmd_args))
                    os.system("%sscripts/test_hnsw %s" % (PROJECT_DIR, cmd_args))
                    # print("numactl --physcpubind=8 --membind=0 %sscripts/test_1 %s" % (PROJECT_DIR, cmd_args))
                    # os.system("numactl --physcpubind=8 --membind=0 %sscripts/test_search %s" % (PROJECT_DIR, cmd_args))
                    y1, y2 = read_file(filename)
                else:
                    y1, y2 = 0, 0
                    pass
            except:
                print(traceback.format_exc())
                y1, y2 = min(self.Y1_record), min(self.Y2_record)
            # configure_index(*filter_index_rule(index_conf))
            # configure_system(filter_system_rule(system_conf))

            # print(f"Parameters changed to: {index_conf} {system_conf}")


            # try:
            #     result = sp.run(f'sudo timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
            #     result = result.stdout.decode().split()
            #     y1, y2 = float(result[-2]), float(result[-3])
            #     # y1, y2 = 1698.4412378836437*(random.random()+0.5), 0.822103*(random.random()+0.5)
            #
            self.Y1_record.append(y1)
            self.Y2_record.append(y2)
            record_path = '/home/yinzh/VDTuner/hnsw/msong/q_10/recordhnsw_10_msong.log'
            # except:
            #     y1, y2 = min(self.Y1_record), min(self.Y2_record)
            y3 = int(time.time()-self.t2)
            self.sampled_times += 1
            self.t2 = time.time()
            print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
            sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {y1} {y2} {y3} >> {record_path}', shell=True, stdout=sp.PIPE)
            # sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {conf_vals} {y1} {y2} {y3} >> {record_path}', shell=True, stdout=sp.PIPE)
            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)

        return np.array([Y1,Y2,Y3]).T


   
    def get_state1(self, knob_vals_arr, batch_size):
        import os, csv, time, subprocess as sp, traceback

        def read_file(_filename):
            perf_file = _filename + "_performance.csv"
            if not os.path.exists(perf_file):
                print("文件不存在")
                return min(self.Y1_record), min(self.Y2_record)
            with open(perf_file, 'r') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                perf = [float(v) for v in first_row[1:]]
                return perf[0], perf[1]
        # def read_file(_filename, n):
        #     """
        #     从 perf_file 读取前 n 行的性能数据，返回两个列表：y1_list, y2_list。
        #     如果文件不存在或行数不足，用 self.Y1_record/ self.Y2_record 的最小值填充。
        #     """
        #     perf_file = _filename + "_performance.csv"
        #     # 历史最差值作为回退
        #     default_y1 = min(self.Y1_record) if self.Y1_record else 0.0
        #     default_y2 = min(self.Y2_record) if self.Y2_record else 0.0

        #     # 如果文件不存在，直接返回全填充
        #     if not os.path.exists(perf_file):
        #         return [default_y1] * n, [default_y2] * n

        #     y1_list, y2_list = [], []
        #     with open(perf_file, 'r') as f:
        #         reader = csv.reader(f)
        #         for i, row in enumerate(reader):
        #             if i >= n:
        #                 break
        #             # 假设每行格式为 [<其他列>, y1, y2, ...]
        #             # 这里取第 2、3 列（索引 1、2）作为 y1, y2
        #             try:
        #                 y1 = float(row[1])
        #                 y2 = float(row[2])
        #             except (IndexError, ValueError):
        #                 # 解析错误时退回默认
        #                 y1, y2 = default_y1, default_y2
        #             y1_list.append(y1)
        #             y2_list.append(y2)

        #     # 如果读到的行不足 n，补齐默认值
        #     if len(y1_list) < n:
        #         missing = n - len(y1_list)
        #         y1_list.extend([default_y1] * missing)
        #         y2_list.extend([default_y2] * missing)

        #     return y1_list, y2_list

        Y1, Y2, Y3 = [], [], []
        # 分批处理配置
        for batch_idx in range(0, len(knob_vals_arr), batch_size):
            batch = knob_vals_arr[batch_idx:batch_idx + batch_size]
            conf_list, batch_log_files = [], []

            # 反归一化并生成日志路径
            for i, record in enumerate(batch):
                try:
                    conf_vals = [self.knob_stand.scale_back(self.names[j], v)[1]
                                for j, v in enumerate(record)]
                    idx_vals = conf_vals[:4]
                    idx_conf = dict(zip(self.names[:4], idx_vals))
                    fname = "_".join(
                        f"{k}_{v}" for k, v in idx_conf.items() if k != 'index_type'
                    )
                    log_path = os.path.join(LOG_DIR, f"{fname}")
                    conf_list.append(conf_vals)
                    batch_log_files.append(log_path)
                except Exception as e:
                    print(f"配置 {i} 生成失败: {e}")
                    conf_list.append(None)
                    batch_log_files.append(None)

            # 筛选有效配置
            valid_confs = [
                (i, c, lf)
                for i, (c, lf) in enumerate(zip(conf_list, batch_log_files))
                if c and lf
            ]
            if not valid_confs:
                continue

            # 基础命令
            cmd = f"{PROJECT_DIR}scripts/test_hnsw {len(valid_confs)}"
            # 拼接每条配置的参数
            # for i, conf_vals, _ in valid_confs:
            #     params_str = " ".join(str(v) for v in conf_vals if v != 'NSG')
            #     cmd += (
            #         f" {PROJECT_DIR}graphknn/graphknn{i+1}"
            #         f" {params_str}"
            #         f" {PROJECT_DIR}graphnsg/graphnsg{i+1}"
            #     )
            # for logf in log_files:
            #     cmd += f" {logf}"
            log_files = [lf for _, _, lf in valid_confs]
            for (i, conf_vals, _), logf in zip(valid_confs, log_files):
                params_str = " ".join(str(v) for v in conf_vals if v != 'hnsw')
                cmd += (
                  
                    f" {params_str}"
                    f" {logf}"
                   
                )
    
           

            # 然后再把 log_files 中的路径追加到命令
            
            # 日志文件参数：合并最后两个为一个
            # log_files = [lf for _, _, lf in valid_confs]
            # if len(log_files) >= 2:
            #     # 合并最后两个日志名
            #     lf1, lf2 = log_files[-2], log_files[-1]
            #     base1 = os.path.splitext(os.path.basename(lf1))[0]
            #     base2 = os.path.splitext(os.path.basename(lf2))[0]
            #     merged_name = f"{base1}_{base2}.log"
            #     merged_path = os.path.join(LOG_DIR, merged_name)
            #     # 保留前面的单独日志，加上合并日志
            #     log_files = log_files[:-2] + [merged_path]
            # # 追加日志参数
            # for logf in log_files:
            #     cmd += f" {logf}"

            # 执行命令
            print(f"执行批次命令: {cmd}")
            os.system(cmd)

            
            y3 = int(time.time() - self.t2)
            self.sampled_times += 1
            self.t2 = time.time()
            record_path = '/home/yinzh/VDTuner/hnsw/msong/q_10/recordhnsw_10_msong.log'
            # 更新采样计数并打印/写日志
            # 逐条配置对应地更新记录
            for (i, conf_vals, _), logf in zip(valid_confs, log_files):
              
                y1, y2 = read_file(logf)
                
                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
                # y3 = int(time.time() - self.t2)
                # self.t2 = time.time()
                # self.sampled_times += 1
                print(f"[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}")
                sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {conf_vals} {y1} {y2} {y3} >> {record_path}', shell=True, stdout=sp.PIPE)
                # sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {y1} {y2} {y3} >> {record_path}', shell=True, stdout=sp.PIPE)
                # sp.run(
                #     f"echo [{self.sampled_times}] {y1} {y2} {y3} >> record1.log",
                #     shell=True, stdout=sp.PIPE
                # )
                Y1.append(y1)
                Y2.append(y2)
                Y3.append(y3)


        return np.array([Y1, Y2, Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

