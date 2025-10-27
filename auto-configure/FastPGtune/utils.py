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

KNOB_PATH = r'/home/yinzh/VDTuner/auto-configure/whole_param_nsg.json'
RUN_ENGINE_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/run_engine.sh'

LOG_DIR = r'/home/yinzh/VDTuner/NSG/msong/q_1/log/'
PROJECT_DIR = r'/home/yinzh/VDTuner/'

DATA_PATH = r"/mnt/data2/zhou/dataset/msong/msong_base.fvecs"
GTRUE = r"/mnt/data2/zhou/dataset/msong/msong_groundtruth.ivecs"
QUERY_PATH = r"/mnt/data2/zhou/dataset/msong/msong_query.fvecs"

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
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(record)]
            # change
            index_value, system_value = conf_value[:5], conf_value[5:]
            index_name, system_name = self.names[:5], self.names[5:]

            index_conf = dict(zip(index_name,index_value))
            system_conf = dict(zip(system_name,system_value))
            try:
                # Y1 为recall，Y2为构图时间
                if index_conf["index_type"] == 'KGraph':
                    if index_conf["S"] > index_conf["L"]:
                        index_conf["S"] = index_conf["L"]
                    filename = "_".join([f"{key}_{value}" for key, value in index_conf.items() if key != "index_type"])
                    log_filename = LOG_DIR + filename + ".log"

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
                elif index_conf["index_type"] == 'NSG':
                    # if index_conf["L"] < index_conf["K"]:
                    #     index_conf["L"] = index_conf["K"]
                    # if index_conf["S"] > index_conf["L"]:
                    #     index_conf["S"] = index_conf["L"]
                    filename ="_".join([f"{k}{v}" for k, v in sorted(index_conf.items()) if k != "index_type"])
                    filename = f"{LOG_DIR}{filename}"
                    cmd = [DATA_PATH]
                    knn_dir = str(PROJECT_DIR +"NSG/msong/q_1/"+"graph" + "knn/" + "knn_" + DATA_SET)
                    cmd_tmp = " ".join([f"{value}" for value in index_conf.values() if value != index_conf["index_type"]])
                    nsg_dir = str(PROJECT_DIR +"NSG/msong/q_1/"+ "graph" + "nsg/" + "nsg_" + DATA_SET)
                    cmd.append(knn_dir)
                    prefix_nums = [100, 100, 8, 10, 100]
                    prefix_strs = [str(x) for x in prefix_nums]
                    cmd.extend(prefix_strs)
                    cmd.append(cmd_tmp)
                    cmd.append(nsg_dir)
                    cmd.append(QUERY_PATH)
                    cmd.append(GTRUE)
                    cmd.append(filename)
                    cmd_args = " ".join(cmd)
                    performance_file = filename + "_performance.csv"
                    if not os.path.exists(performance_file):
                        print("%sscripts/test_1 %s" % (PROJECT_DIR, cmd_args))
                        os.system("%sscripts/test_1 %s" % (PROJECT_DIR, cmd_args))
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
            # except:
            #     y1, y2 = min(self.Y1_record), min(self.Y2_record)
            y3 = int(time.time()-self.t2)
            self.sampled_times += 1
            self.t2 = time.time()
            record_path = '/home/yinzh/VDTuner/NSG/msong/q_1/recordNSG_1_msong.log'
            print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
            sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {y1} {y2} {y3} >> {record_path}', shell=True, stdout=sp.PIPE)

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)

        return np.array([Y1,Y2,Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

