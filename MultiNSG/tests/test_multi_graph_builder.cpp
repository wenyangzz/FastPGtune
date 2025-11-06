//
// 多图构建测试程序
//

#include <efanna2e/multi_graph_builder.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <efanna2e/index_random.h>
#include <efanna2e/index_graph.h>
void read_fvecs(char* filename, float*& data, unsigned& num, unsigned& dim){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){
        std::cerr << "open file error!" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)fsize / (dim + 1) / 4;
    #ifdef __GNUC__
    #ifdef __AVX__
    #define DATA_ALIGN_FACTOR 8
    #else
    #ifdef __SSE2__
    #define DATA_ALIGN_FACTOR 4
    #else
    #define DATA_ALIGN_FACTOR 1
    #endif
    #endif
    #endif
    int a = 16;
    unsigned new_dim = (dim + a - 1) / a * a;
    data = new float[(size_t)num * (size_t)new_dim];
    memset(data, 0, sizeof(float) * (size_t)num * (size_t)new_dim);
    in.seekg(0, std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * new_dim), dim * 4);
    }
    in.close();
    dim = new_dim;
}
void load_fvecs(const char* filename,
                float*& data,
                unsigned& num,
                unsigned& dim) 
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) { 
    std::cerr << "open file error: " << filename << std::endl; 
    exit(-1);
  }

  // 先读第一条记录的维度
  uint32_t d;
  in.read(reinterpret_cast<char*>(&d), sizeof(d));
  dim = d;

  // 用一个 vector 暂存所有值
  std::vector<float> buffer;
  buffer.reserve(1024 * dim);  // 先给个大致预留

  // 读完第一条向量的数据
  {
    std::vector<float> tmp(dim);
    in.read(reinterpret_cast<char*>(tmp.data()), sizeof(float) * dim);
    buffer.insert(buffer.end(), tmp.begin(), tmp.end());
  }

  // 继续读后面的记录，直到文件结束
  while (true) {
    // 再读一个维度字段
    if (!in.read(reinterpret_cast<char*>(&d), sizeof(d)))
      break;   // 到文件尾了
    if (d != dim) {
      std::cerr << "inconsistent dim, got " << d
                << " expected " << dim << std::endl;
      exit(-1);
    }
    // 读这一条的 dim floats
    std::vector<float> tmp(dim);
    in.read(reinterpret_cast<char*>(tmp.data()), sizeof(float) * dim);
    buffer.insert(buffer.end(), tmp.begin(), tmp.end());
  }
  in.close();

  // 把 buffer 拷贝给 caller
  num = buffer.size() / dim;
  data = new float[buffer.size()];
  std::memcpy(data, buffer.data(), buffer.size() * sizeof(float));
}

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}
void load_ivecs(const char* filename, std::vector<std::vector<unsigned>>& result){
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
    std::cout << "open file error" << std::endl;
  }
  unsigned K;
  in.read((char*)&K, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  unsigned num = (unsigned)(fsize / (K + 1) / 4);
  in.seekg(0, std::ios::beg);
  result.reserve(num);
  result.resize(num);
  unsigned kk = (K + 3) / 4 * 4;
  for(unsigned i = 0; i < num; i++){
    in.seekg(4, std::ios::cur);
    result[i].resize(K);
    result[i].reserve(kk);
    in.read((char*)result[i].data(), K * sizeof(unsigned));
  }
  in.close();
}

void SetKnnParams(efanna2e::Parameters& paras, int baseid, int argc, char** argv) {
  unsigned K = (unsigned)atoi(argv[baseid + 0]);
  unsigned L = (unsigned)atoi(argv[baseid + 1]);
  unsigned iter = (unsigned)atoi(argv[baseid + 2]);
  unsigned S = (unsigned)atoi(argv[baseid + 3]);
  unsigned R = (unsigned)atoi(argv[baseid + 4]);

  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);
}

void SetNsgParams(efanna2e::Parameters& paras, int baseid, int argc, char** argv) {
  unsigned nsg_L = (unsigned)atoi(argv[baseid]);
  unsigned nsg_R = (unsigned)atoi(argv[baseid + 1]);
  unsigned nsg_C = (unsigned)atoi(argv[baseid + 2]);
  unsigned search_L = (unsigned)atoi(argv[baseid + 3]);
  std::string performance_log(argv[baseid + 4]);
  paras.Set<unsigned>("L", nsg_L);
  paras.Set<unsigned>("R", nsg_R);
  paras.Set<unsigned>("C", nsg_C);
  paras.Set<std::string>("performance_log", performance_log);
  paras.Set<unsigned>("Search_L", search_L);
  std::cout << "  参数: L=" << nsg_L << ", R=" << nsg_R << ", C=" << nsg_C 
            << ", search_L=" << search_L << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: "<< argv[0] << " data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C search_L performance_log nsg_graph] query_file groundtruth_file" << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  
  unsigned points_num, dim;
  read_fvecs(argv[1], data_load, points_num, dim);
  std::cout <<points_num <<"个点" << std::endl;
  std::cout <<dim <<"维度" << std::endl;
  
  int num_graphs = atoi(argv[2]);
  if (argc != 3 + num_graphs * 12 + 2) {
    std::cout << "Usage: "<< argv[0] << " data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C search_L performance_log nsg_graph] query_file groundtruth_file" << std::endl;
    std::cout << "argc" << argc << std::endl;
    exit(-1);
  }
  efanna2e::MultiGraphBuilder builder(dim, points_num, efanna2e::L2);
  auto a = std::chrono::high_resolution_clock::now();
  //knn 不一样的情况
  // {
  //   for (int i = 0; i < num_graphs; i++){
  //     efanna2e::IndexRandom init_index(dim, points_num);
  //     efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
  //     int base_idx = 3 + i * 12;
  //     char* graph_filename = argv[base_idx];
  //     efanna2e::Parameters knn_params, nsg_params;
  //     SetKnnParams(knn_params, base_idx + 1, argc, argv);
  //     SetNsgParams(nsg_params, base_idx + 6, argc, argv);
  //     std::string save_graph_path(argv[base_idx + 11]);
  //     index.Build(points_num, data_load, knn_params);
  //     index.Save(graph_filename);
  //     // using index.final_graph
  //     // no load knn
  //     builder.AddGraphConfig(nsg_params, save_graph_path, index.ExtractFinalGraph());
  //   }
  // }
  //所有的knn都一样
    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
    efanna2e::Parameters knn_params, nsg_params;
    SetKnnParams(knn_params, 3 + 1, argc, argv);
    index.Build(points_num, data_load, knn_params);
    char* graph_filename = argv[3];
    index.Save(graph_filename);
    for (int i = 0; i < num_graphs; i++){
        // efanna2e::IndexRandom init_index(dim, points_num);
        // efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
        int base_idx = 3 + i * 12;
        // char* graph_filename = argv[base_idx];
        efanna2e::Parameters nsg_params;
        // SetKnnParams(knn_params, base_idx + 1, argc, argv);
        SetNsgParams(nsg_params, base_idx + 6, argc, argv);
        std::string save_graph_path(argv[base_idx + 11]);
        // index.Build(points_num, data_load, knn_params);
        // index.Save(graph_filename);
        // using index.final_graph
        // no load knn
        builder.AddGraphConfig(nsg_params, save_graph_path, index.ExtractFinalGraphCopy());
      }
  auto a2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_knn = a2 - a;
  std::cout << "knn总耗时: " << diff_knn.count() << " 秒" << std::endl;
  std::cout<<"omp_get_max_threads() " <<omp_get_max_threads()<<std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  // 构建所有图
  std::cout << std::endl << "开始构建 " << num_graphs << " 个图..." << std::endl;
  std::pair<unsigned long long, unsigned long long> cmp;
  cmp = builder.BuildAllGraphs(data_load);
  unsigned long long dist_compute = cmp.first;
  unsigned long long prune_compute = cmp.second;
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  const std::string log_file = 
      "/home/yinzh/MultiNSG_pruner/MultiNSG/tests/prun_n/computer_dist1.log";
  const std::string log_file2 = 
      "/home/yinzh/MultiNSG_pruner/MultiNSG/tests/prun_n/computer_prune1.log";    
  const std::string log_file1 = 
      "/home/yinzh/MultiNSG_pruner/MultiNSG/tests/prun_n/index_time1.log";
  std::cout << "准备写入 dist_compute 日志..." << std::endl;
  std::ofstream ofs1(log_file, std::ios::out | std::ios::app);
  if (!ofs1) {
      std::cerr << "无法打开日志文件: " << log_file << std::endl;
  } else {
      ofs1 << "dist_compute: " << dist_compute << "\n";
      ofs1.close();
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  std::cout << "dist_compute:  " << dist_compute<< "\n";
  std::cout << "准备写入 prune_compute  日志..." << std::endl;
  std::ofstream ofs3(log_file2, std::ios::out | std::ios::app);
  if (!ofs3) {
      std::cerr << "无法打开日志文件: " << log_file2 << std::endl;
  } else {
      ofs3 << "prune_compute: " << prune_compute << "\n";
      ofs3.close();
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  std::cout << "prune_compute:  " << prune_compute<< "\n";
  std::ofstream ofs2(log_file1, std::ios::out | std::ios::app);
  if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
  } else {
      ofs2 << "IndexKNN time: " << diff_knn.count()<< "s\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
  } else {
      ofs2 << "Indexing time: " << diff.count()<< "s\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  std::cout << "多图构建总耗时: " << diff.count() << " 秒" << std::endl;
  //search
  {
    unsigned base_id = 3 + num_graphs * 12;
    float* query_data = nullptr;
    unsigned query_num;
    read_fvecs(argv[base_id], query_data, query_num, dim);
    std::cout <<query_num <<"个点" << std::endl;
    std::vector<std::vector<unsigned>> gtrue;
    load_ivecs(argv[base_id + 1], gtrue);
    // std::vector<unsigned> L_values = {60, 80, 100, 120, 140, 160, 180, 200};
    // builder.EvaluateGraphs(query_data, query_num, 100, L_values,
    //                       gtrue, argv[base_id + 2]);
    auto s2 = std::chrono::high_resolution_clock::now();
    builder.EvaluateGraphs(query_data, query_num, 10, gtrue);
    std::chrono::duration<double> diff2 = std::chrono::high_resolution_clock::now() - s2;
    if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
    } else {
        ofs2 << "search time: " << diff2.count()<< "s\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
  }
  // delete[] data_load;
  return 0;
}