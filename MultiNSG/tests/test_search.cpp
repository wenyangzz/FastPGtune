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

#define REPEAT_COUNT 10

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

int main(int argc, char** argv) {
    if (argc != 16) {
    std::cout << "data_file nn_file K L iter S R nsg_L nsg_R nsg_C nsg_graph query_file groundtruth_file log_file" << std::endl;
    exit(-1);
    }
    float* data_load = NULL;

    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);

    // omp_set_num_threads(omp_get_max_threads()/2);
  // knn
    // auto graph_filename = argv[2];
    // auto K = (unsigned)atoi(argv[3]);
    // auto L = (unsigned)atoi(argv[4]);
    // auto iter = (unsigned)atoi(argv[5]);
    // auto S = (unsigned)atoi(argv[6]);
    // auto R = (unsigned)atoi(argv[7]);
    // efanna2e::IndexRandom init_index(dim, points_num);
    // efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
    // efanna2e::Parameters paras;
    // paras.Set<unsigned>("K", K);
    // paras.Set<unsigned>("L", L);
    // paras.Set<unsigned>("iter", iter);
    // paras.Set<unsigned>("S", S);
    // paras.Set<unsigned>("R", R);
    // index.Build(points_num, data_load, paras);
    // index.Save(graph_filename);
    auto nsg_filename = argv[12];
    // auto nsg_L = (unsigned)atoi(argv[8]);
    // auto nsg_R = (unsigned)atoi(argv[9]);
    // auto nsg_C = (unsigned)atoi(argv[10]);
    efanna2e::IndexNSG index_nsg(dim, points_num, efanna2e::L2, nullptr);
    // efanna2e::Parameters paras_nsg;
    // paras_nsg.Set<unsigned>("L", nsg_L);
    // paras_nsg.Set<unsigned>("R", nsg_R);
    // paras_nsg.Set<unsigned>("C", nsg_C);
    // paras_nsg.Set<std::string>("nn_graph_path", graph_filename);
    // index_nsg.Build(points_num, data_load, paras_nsg);
    // index_nsg.Save(nsg_filename);
    index_nsg.Load(nsg_filename);
  {
    float* query_data = nullptr;
    unsigned query_num;
    load_data(argv[13], query_data, query_num, dim);

    std::vector<std::vector<unsigned>> gtrue;
    load_ivecs(argv[14], gtrue);
    unsigned L = atoi(argv[11]);
    std::vector<unsigned> L_values = {18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 60, 70, 90, 100, 120};
    // std::vector<unsigned> L_values = {90};
    EvaluateGraph(query_data, data_load, query_num, dim, 10, gtrue, index_nsg, L_values, argv[15]);
  }
  delete[] data_load;
  return 0;
}


