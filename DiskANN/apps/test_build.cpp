#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "index.h"           // DiskANN 相关头
#include "index_factory.h"
#include "utils.h"
#include "abstract_index.h" 
#include <chrono> // 添加时间测量头文件
#include "dist_list_pool.h" // 添加dist
namespace po = boost::program_options;

struct GraphSpec {
    uint32_t    R;
    uint32_t    L;
    float       alpha;
    uint32_t    search_L; 
    std::string graph_path;   // 单个搜索深度
    std::string perf_prefix;
};

// 独立的内存统计函数实现
#ifdef __linux__
#include <unistd.h>
#include <cstdio>
size_t getCurrentRSS() {
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp == nullptr) return 0;
    
    long rss = 0;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    
    long page_size = sysconf(_SC_PAGESIZE);
    return rss * page_size;
}
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
size_t getCurrentRSS() {
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
}
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
size_t getCurrentRSS() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;
    return (size_t)info.resident_size;
}
#else
size_t getCurrentRSS() { 
    return 0;  // 不支持的平台
}
#endif
// 测试并把性能写入 CSV（仅针对单个 L）
void test_search_performance(const std::string &index_path,
                             const std::string &query_file,
                             const std::string &truthset_file,
                             uint32_t          K,
                             uint32_t          L,
                             uint32_t          num_threads,
                             const std::string &perf_prefix) {
    // 1) 加载查询向量
    float *queries = nullptr;
    size_t qnum, qdim, qaligned;
    diskann::load_aligned_bin<float>(query_file, queries, qnum, qdim, qaligned);

    // 2) 加载真值
    uint32_t *gt_ids = nullptr;
    float    *gt_dists = nullptr;
    size_t    gt_num, gt_dim;
    bool      do_recall = false;
    if (truthset_file != "null" && file_exists(truthset_file)) {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        do_recall = (gt_num == qnum);
    }
    // 3) 构建 config 并 load
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(diskann::Metric::L2)
                      .with_dimension(qdim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type("float")
                      .build();
    auto idx = diskann::IndexFactory(config).create_instance();
    idx->load(index_path.c_str(), num_threads, L);

    std::vector<uint32_t> result_ids(K * qnum);

    // 4) 打开 CSV，写表头
    std::string csv_file = perf_prefix + "_performance.csv";
    std::ofstream ofs(csv_file);
    // ofs << "L,QPS,AvgLatency_us,Recall@K\n";

    // 5) 并行搜索测时
    int num_runs = 5;

    // 用于累加
    double sum_qps    = 0.0;
    double sum_recall = 0.0;

    for (int run = 1; run <= num_runs; ++run) {
        // 4) 并行搜索测时
        omp_set_num_threads(num_threads);
        auto t0 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic,1)
        for (int64_t i = 0; i < (int64_t)qnum; ++i) {
            idx->search(queries + i * qaligned, K, L, result_ids.data() + i * K);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(t1 - t0).count();
        double qps       = qnum / total_sec;

        // 5) 计算 Recall@K
        double recall = 0.0;
        if (do_recall) {
            recall = diskann::calculate_recall(
                static_cast<uint32_t>(qnum),
                gt_ids, gt_dists, static_cast<uint32_t>(gt_dim),
                result_ids.data(),
                K, K);
        }

        // 6) 累加到总和
        sum_qps    += qps;
        sum_recall += recall;
    }

    // 8) 计算平均并写入
    double avg_qps    = sum_qps    / num_runs;
    double avg_recall = sum_recall / num_runs;
    

    // 7) 写入 CSV 并结束
    ofs << L << ","
        << std::fixed << std::setprecision(4) << avg_recall << ","
        << std::fixed << std::setprecision(4) << avg_qps << "\n";
    ofs.close();

    std::cout << "Wrote performance to " << csv_file << "\n";

    diskann::aligned_free(queries);
    if (do_recall) {
        delete[] gt_ids;
        delete[] gt_dists;
    }
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法： " << argv[0]
                  << " <data_file> <num_graph>"
                     " [-R  -L  -alpha -search_l <graph_file> <perf_prefix>]...\n";
        return -1;
    }
    std::string data_path = argv[1];
    uint32_t num_graph  = static_cast<uint32_t>(std::stoul(argv[2]));

    // 每组期望 6 个参数：-R, -L, -alpha, -search_l, graph_file,perf_prefix
    const int tokens_per_graph = 6;
    std::vector<std::string> args(argv + 3, argv + argc);
    if ((int)args.size() != (int)num_graph * tokens_per_graph) {
        std::cerr << "错误：解析到 " << args.size() 
                  << " 个参数，但期望 " 
                  << num_graph * tokens_per_graph << " 个。每组期望 6 个参数：-R, -L, -alpha, -search_l, graph_file,perf_prefix\n";
        return -1;
    }

    // 解析每一组参数
    std::vector<GraphSpec> specs;
    specs.reserve(num_graph);
    for (uint32_t i = 0; i < num_graph; ++i) {
        size_t b = i * tokens_per_graph;
        uint32_t    R    = static_cast<uint32_t>(std::stoul(args[b]));
        uint32_t    L    = static_cast<uint32_t>(std::stoul(args[b+1]));
        float       a    = std::stof(args[b+2]);
        uint32_t    search_l   = static_cast<uint32_t>(std::stoul(args[b+3]));
        std::string path_graph = args[b+4];
        std::string perf_prefix = args[b+5];
        specs.push_back({R, L, a, search_l,path_graph,perf_prefix});
    }

    // 获取数据规模
    size_t data_num, data_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);
    // size_t memory_before_build = getCurrentRSS();
    // std::cout << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB" << std::endl;
    
    // const std::string memory_log_file_1 = 
    //     "/home/yinzh/DiskANN/Vamana_n/memory_usage_1.log";
    // std::ofstream memory_ofs_1(memory_log_file_1, std::ios::out | std::ios::app);
    // if (memory_ofs_1) {
    //     memory_ofs_1 << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB\n";
    // } else {
    //     std::cerr << "无法打开内存日志文件: " << memory_before_build << std::endl;
    // }
     // 先创建所有实例，存入 vector<AbstractIndex>
    std::vector<std::unique_ptr<diskann::AbstractIndex>> indices;
    indices.reserve(num_graph);
    std::vector<uint32_t> visit_order;
    std::unique_ptr<diskann::DistListPool> dist_list_pool_;
    dist_list_pool_ = std::make_unique<diskann::DistListPool>(100, data_num);
    unsigned long long dist_compute = 0;
    unsigned long long prune_compute = 0;
    for (uint32_t i = 0; i < num_graph; ++i) {
        const auto &sp = specs[i];

        // 构建写参数
        auto write_params = diskann::IndexWriteParametersBuilder(sp.L, sp.R)
                                .with_alpha(sp.alpha)
                                .with_num_threads(omp_get_max_threads())
                                .build();

        // 构建索引配置
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(diskann::Metric::L2)    // 可选 INNER_PRODUCT/COSINE
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type("float")             // 根据实际数据类型调整
                          .with_index_write_params(write_params)
                          .build();
        
        // 创建实例并保存到 vector
        auto idx = diskann::IndexFactory(config).create_instance();
        indices.emplace_back(std::move(idx));
    }
    
    // 2) 对每个实例分别 build 并 save
    for (uint32_t i = 0; i < num_graph; ++i) {
        auto &idx       = indices[i];
        const auto &sp  = specs[i];
        auto filter_params = diskann::IndexFilterParamsBuilder()
                                    .with_save_path_prefix(sp.graph_path.c_str())
                                    .build();     
        std::cout << "Building graph[" << i << "] with R=" << sp.R
                  << " L=" << sp.L << " alpha=" << sp.alpha << " …\n";

        // 构建
        idx->build(data_path, data_num, filter_params);
        //  size_t memory_before_build = getCurrentRSS();
        // std::cout << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB" << std::endl;
        
        // const std::string memory_log_file_1 = 
        //     "/home/yinzh/DiskANN/Vamana_n/memory_usage_1.log";
        // std::ofstream memory_ofs_1(memory_log_file_1, std::ios::out | std::ios::app);
        // if (memory_ofs_1) {
        //     memory_ofs_1 << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB\n";
        // } else {
        //     std::cerr << "无法打开内存日志文件: " << memory_before_build << std::endl;
        // }
        idx->initializeProcessing(visit_order);        
    }

    size_t memory_before_build = getCurrentRSS();
    std::cout << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB" << std::endl;
    
    const std::string memory_log_file_1 = 
        "/home/yinzh/DiskANN/Vamana_n/memory_usage_1.log";
    std::ofstream memory_ofs_1(memory_log_file_1, std::ios::out | std::ios::app);
    if (memory_ofs_1) {
        memory_ofs_1 << "构图完成前的内存使用: " << memory_before_build / (1000 * 1000) << " MB\n";
    } else {
        std::cerr << "无法打开内存日志文件: " << memory_before_build << std::endl;
    }
    auto s = std::chrono::high_resolution_clock::now();
    //并行构图
    // omp_set_num_threads(36);
    //  int num_threads = omp_get_max_threads();
    //     std::cout << "Threads "  << num_threads << std::endl;
    #pragma omp parallel for schedule(dynamic, 2048) reduction(+: dist_compute,prune_compute)
                for (int64_t node_ctr = 0; node_ctr < static_cast<int64_t>(visit_order.size()); node_ctr++) {
                    diskann::DistList* dl = dist_list_pool_->getFreeVisitedList();
                    for (uint32_t i = 0; i < num_graph; ++i) {
                            auto &idx       = indices[i];
                            auto cmps = idx->add_point1(visit_order, dl, node_ctr);
                            dist_compute += cmps.second;
                            prune_compute += cmps.first;
                    }
                    dist_list_pool_->releaseDistList(dl);
                }
    for (uint32_t i = 0; i < num_graph; ++i) {
        auto &idx       = indices[i];
        const auto &sp  = specs[i];
        prune_compute+=idx->globalOptimizationPrune(visit_order);
        idx->end();
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Indexing time: " << diff.count()<< "\n";
      // ==================== 构图完成后的内存统计 ====================
    size_t memory_after_build = getCurrentRSS();
    std::cout << "构图完成后的内存使用: " << memory_after_build / (1000 * 1000) << " MB" << std::endl;
    
    const std::string memory_log_file = 
        "/home/yinzh/DiskANN/Vamana_n/memory_usage.log";
    std::ofstream memory_ofs(memory_log_file, std::ios::out | std::ios::app);
    if (memory_ofs) {
        memory_ofs << "构图完成后的内存使用: " << memory_after_build / (1000 * 1000) << " MB\n";
    } else {
        std::cerr << "无法打开内存日志文件: " << memory_log_file << std::endl;
    }
    const std::string log_file = 
        "/home/yinzh/DiskANN/Vamana_n/computer_dist.log";
    const std::string log_file1 = 
        "/home/yinzh/DiskANN/Vamana_n/index_time.log";
    const std::string log_file2 = 
        "/home/yinzh/DiskANN/Vamana_n/computer_prune.log";    
    std::ofstream ofs1(log_file, std::ios::out | std::ios::app);
    if (!ofs1) {
        std::cerr << "无法打开日志文件: " << log_file << std::endl;
    } else {
        ofs1 << "dist_compute: " << dist_compute << "\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
    std::cout << "dist_compute:  " << dist_compute<< "\n";
    std::ofstream ofs3(log_file2, std::ios::out | std::ios::app);
    if (!ofs3) {
        std::cerr << "无法打开日志文件: " << log_file << std::endl;
    } else {
        ofs3 << "computer_prune: " << prune_compute << "\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
    std::cout << "computer_prune:  " << prune_compute<< "\n";
    std::ofstream ofs2(log_file1, std::ios::out | std::ios::app);
    if (!ofs2) {
        std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
    } else {
        ofs2 << "Indexing time: " << diff.count()<< "s\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
    // 保存
    auto save_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_graph; ++i) {
        auto &idx       = indices[i];
        const auto &sp  = specs[i];
        idx->save(sp.graph_path.c_str());
        std::cout << "Saved graph[" << i << "] to " << sp.graph_path << "\n";    
    }
    std::chrono::duration<double> diff_save = std::chrono::high_resolution_clock::now() - save_time;
    if (!ofs2) {
        std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
    } else {
        ofs2 << "Search time: " << diff_save.count()<< "s\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
    const std::string truthset_file = 
        "/home/yinzh/DiskANN/build/data/sift/sift_query_learn_gt100";
    const std::string query_file = "/home/yinzh/DiskANN/build/data/sift/sift_query.fbin";
    auto search_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_graph; ++i) {
        const auto &sp  = specs[i];
        test_search_performance(sp.graph_path,
                             query_file,
                             truthset_file,
                             10,
                             sp.search_L,
                             70,
                             sp.perf_prefix) ;   
    }
    std::chrono::duration<double> diff_search = std::chrono::high_resolution_clock::now() - search_time;
    if (!ofs2) {
        std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
    } else {
        ofs2 << "Save time: " << diff_search.count()<< "s\n";
        // ofs.close(); // ofstream 在析构时会自动关闭
    }
    
    return 0;
}
