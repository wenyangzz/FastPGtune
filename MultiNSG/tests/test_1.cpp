#include <efanna2e/multi_graph_builder.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <efanna2e/index_random.h>
#include <efanna2e/index_graph.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#define REPEAT_COUNT 5

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

float ComputeRecall(const std::vector<std::vector<unsigned>>& gtrue,
  const std::vector<std::vector<unsigned>>& results,
  size_t K) {
    unsigned query_num = results.size();
    double total_recall = 0.0;
    for(unsigned i = 0; i < query_num; i++){
      unsigned matches = 0;
      std::unordered_set<unsigned> gtrue_set(gtrue[i].begin(), gtrue[i].begin() + K);
      for(size_t j = 0; j < K; j++){
        if(gtrue_set.find(results[i][j]) != gtrue_set.end()){
          matches++;
        }
      }
      total_recall += static_cast<double>(matches) / K;
    }
    return query_num > 0 ? total_recall / query_num : 0.0f;
}

void EvaluateGraph(const float* query_data, const float* data_load, unsigned query_num, 
  unsigned dim, unsigned K, const std::vector<std::vector<unsigned>>& gtrue, 
  efanna2e::IndexNSG& index_nsg, const std::vector<unsigned>& L_values,
  const char* output_file){
  std::vector<double> run_times(REPEAT_COUNT);
  std::string structure_filename = std::string(output_file) + "_structure.csv";
  std::string performance_filename = std::string(output_file) + "_performance.csv";
  std::ofstream structure_out(structure_filename);
  std::ofstream performance_out(performance_filename);
  std::vector<std::vector<unsigned>> search_result(query_num, std::vector<unsigned>(K));
  double total_time = 0.0;
  {
    size_t nd_ = index_nsg.GetSizeOfDataset();
    const std::vector<std::vector<unsigned>>* final_graph = index_nsg.GetFinalGraph();
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::variance>> acc;
    double deg_variance = 0.0, avg_degree = 0.0;
    unsigned max_degree = 0;
    for (size_t i = 0; i < nd_; i++) {
        unsigned degree = final_graph->at(i).size();
        acc(degree);
        max_degree = std::max(max_degree, degree);
    }
    avg_degree = boost::accumulators::mean(acc);
    deg_variance = boost::accumulators::variance(acc);
    structure_out << avg_degree << "," 
                  << deg_variance << ","
                  << max_degree << std::endl;
  }
  for (const auto& L : L_values){
    total_time = 0;
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);
    for(int j = 0; j < REPEAT_COUNT; j++){
      auto s = std::chrono::high_resolution_clock::now();
      for (unsigned i = 0; i < query_num; i++) {
        index_nsg.Search(query_data + i * dim, data_load, K, paras, search_result[i].data());
      }
      auto e = std::chrono::high_resolution_clock::now();
      run_times[j] = std::chrono::duration<double>(e - s).count();
    }
    for(int run = 1; run < REPEAT_COUNT; run++){
      total_time += run_times[run];
    }
    double qps = query_num / (total_time / (REPEAT_COUNT - 1));
    float recall = ComputeRecall(gtrue, search_result, K);
    performance_out << L << "," << recall << "," << qps << std::endl;
  }
  performance_out.close();
  structure_out.close();
}

int main(int argc, char **argv)
{
  if (argc != 16)
  {
    std::cout << "data_file nn_file K L iter S R nsg_L nsg_R nsg_C nsg_graph query_file groundtruth_file log_file" << std::endl;
    exit(-1);
  }
  float *data_load = NULL;
  float *data_load_1 = NULL;
  unsigned points_num, dim;
  read_fvecs(argv[1], data_load, points_num, dim);
  std::cout <<points_num <<"个点" << std::endl;
  std::cout <<dim <<"维度" << std::endl;
  float *query_data = nullptr;
  unsigned query_num;
  // read_fvecs(argv[13], query_data, query_num, dim);
  std::cout <<query_num <<"个点" << std::endl;
  std::cout <<dim <<"维度" << std::endl;
  // omp_set_num_threads(omp_get_max_threads()/2);
  // knn
  auto graph_filename = argv[2];
  auto K = (unsigned)atoi(argv[3]);
  auto L = (unsigned)atoi(argv[4]);
  auto iter = (unsigned)atoi(argv[5]);
  auto S = (unsigned)atoi(argv[6]);
  auto R = (unsigned)atoi(argv[7]);
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *)(&init_index));
  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);
  auto a = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  index.Save(graph_filename);
  std::chrono::duration<double> diff_knn = std::chrono::high_resolution_clock::now() - a;
  std::cout << "Indexknn time: " << diff_knn.count()<< "\n";
  auto nsg_filename = argv[12];
  auto nsg_L = (unsigned)atoi(argv[8]);
  auto nsg_R = (unsigned)atoi(argv[9]);
  auto nsg_C = (unsigned)atoi(argv[10]);
  efanna2e::IndexNSG index_nsg(dim, points_num, efanna2e::L2, nullptr);
  efanna2e::Parameters paras_nsg;
  paras_nsg.Set<unsigned>("L", nsg_L);
  paras_nsg.Set<unsigned>("R", nsg_R);
  paras_nsg.Set<unsigned>("C", nsg_C);
  paras_nsg.Set<std::string>("nn_graph_path", graph_filename);
  // read_fvecs(argv[1], data_load_1, points_num, dim);
  // read_fvecs(argv[13], query_data, query_num, dim);
  // data_load = efanna2e::data_align(data_load, points_num, dim);
  std::cout <<dim <<"维度" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  unsigned long long dist_compute = index_nsg.Build(points_num, data_load, paras_nsg);
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
  std::cout << "Indexing time: " << diff.count()<< "\n";
  const std::string log_file = 
      "/home/yinzh/VDTuner/NSG/glove/dantu/computer_dist1.log";
  const std::string log_file1 = 
      "/home/yinzh/VDTuner/NSG/glove/dantu/index_time1.log";
  std::ofstream ofs1(log_file, std::ios::out | std::ios::app);
  if (!ofs1) {
      std::cerr << "无法打开日志文件: " << log_file << std::endl;
  } else {
      ofs1 << "dist_compute: " << dist_compute << "\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  std::cout << "dist_compute:  " << dist_compute<< "\n";
  std::ofstream ofs2(log_file1, std::ios::out | std::ios::app);
  if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
  } else {
      ofs2 << "Indexknn time: " << diff_knn.count()<< "s\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
  } else {
      ofs2 << "Indexing time: " << diff.count()<< "s\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  auto s1 = std::chrono::high_resolution_clock::now();
  index_nsg.Save(nsg_filename);
  std::chrono::duration<double> diff1 = std::chrono::high_resolution_clock::now() - s1;
  if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
  } else {
      ofs2 << "Save time: " << diff1.count()<< "s\n";
      // ofs.close(); // ofstream 在析构时会自动关闭
  }
  {
    float *query_data = nullptr;
    unsigned query_num;
    unsigned query_dim;
    read_fvecs(argv[13], query_data, query_num, query_dim);
    // query_data = efanna2e::data_align(query_data,query_num, query_dim);
    std::cout <<query_dim <<"维度" << std::endl;
    std::vector<std::vector<unsigned>> gtrue;
    load_ivecs(argv[14], gtrue);
    unsigned L = atoi(argv[11]);
    std::vector<unsigned> L_values = {10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600};
    auto s2 = std::chrono::high_resolution_clock::now();
    index_nsg.EvaluateGraph(query_data, query_num, 10, gtrue, L_values, argv[15]);
    std::chrono::duration<double> diff2 = std::chrono::high_resolution_clock::now() - s2;
    if (!ofs2) {
      std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
    } else {
        ofs2 << "search time: " << diff2.count()<< "s\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
  }
  delete[] data_load;
  return 0;
}
