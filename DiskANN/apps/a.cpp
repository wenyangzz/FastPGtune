#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <set>
#include <string>

struct GraphHeader {
    size_t expected_file_size;
    uint32_t max_observed_degree;
    uint32_t start;
    size_t file_frozen_pts;
};

GraphHeader read_graph_header(const std::string& filename) {
    GraphHeader header;
    std::ifstream in(filename, std::ios::binary);
    
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    in.read(reinterpret_cast<char*>(&header.expected_file_size), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&header.max_observed_degree), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&header.start), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&header.file_frozen_pts), sizeof(size_t));
    
    if (!in) {
        std::cerr << "Error: Failed to read header from " << filename << std::endl;
        exit(1);
    }
    
    std::cout << "File: " << filename << std::endl;
    std::cout << "  Expected file size: " << header.expected_file_size << std::endl;
    std::cout << "  Max observed degree: " << header.max_observed_degree << std::endl;
    std::cout << "  Start node: " << header.start << std::endl;
    std::cout << "  Frozen points: " << header.file_frozen_pts << std::endl;
    
    return header;
}

double calculate_same_neighbors_from_files(const std::string& file1, 
                                          const std::string& file2, 
                                          size_t num_nodes = 100000) {
    std::ifstream in1(file1, std::ios::binary);
    std::ifstream in2(file2, std::ios::binary);
    
    if (!in1.is_open()) {
        std::cerr << "Error: Cannot open file " << file1 << std::endl;
        return -1.0;
    }
    if (!in2.is_open()) {
        std::cerr << "Error: Cannot open file " << file2 << std::endl;
        return -1.0;
    }
    
    std::cout << "=== Comparing Graphs ===" << std::endl;
    std::cout << "Graph 1: " << file1 << std::endl;
    std::cout << "Graph 2: " << file2 << std::endl;
    std::cout << "Nodes to compare: " << num_nodes << std::endl;
    std::cout << "=========================" << std::endl;
    
    // 读取文件头
    GraphHeader header1 = read_graph_header(file1);
    GraphHeader header2 = read_graph_header(file2);
    
    // 跳过文件头
    size_t header_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
    in1.seekg(header_size, std::ios::beg);
    in2.seekg(header_size, std::ios::beg);
    
    size_t total_same_neighbors = 0;
    size_t total_neighbors_in_graph1 = 0;
    size_t nodes_processed = 0;
    
    // 检查文件是否有足够节点
    size_t max_possible_nodes1 = (header1.expected_file_size - header_size) / (sizeof(uint32_t) * (header1.max_observed_degree + 1));
    size_t max_possible_nodes2 = (header2.expected_file_size - header_size) / (sizeof(uint32_t) * (header2.max_observed_degree + 1));
    
    size_t actual_nodes_to_process = std::min(num_nodes, std::min(max_possible_nodes1, max_possible_nodes2));
    std::cout << "Actual nodes to process: " << actual_nodes_to_process << std::endl;
    
    for (size_t i = 0; i < actual_nodes_to_process; i++) {
        // 检查文件读取状态
        if (!in1 || !in2) {
            std::cout << "Warning: Reached end of file at node " << i << std::endl;
            break;
        }
        
        // 读取图1的节点邻居
        uint32_t k1;
        in1.read(reinterpret_cast<char*>(&k1), sizeof(uint32_t));
        if (!in1) break;
        
        std::vector<uint32_t> neighbors1(k1);
        if (k1 > 0) {
            in1.read(reinterpret_cast<char*>(neighbors1.data()), k1 * sizeof(uint32_t));
            if (!in1) break;
        }
        
        // 读取图2的节点邻居
        uint32_t k2;
        in2.read(reinterpret_cast<char*>(&k2), sizeof(uint32_t));
        if (!in2) break;
        
        std::vector<uint32_t> neighbors2(k2);
        if (k2 > 0) {
            in2.read(reinterpret_cast<char*>(neighbors2.data()), k2 * sizeof(uint32_t));
            if (!in2) break;
        }
        
        // 如果图1中该节点没有邻居，跳过
        if (neighbors1.empty()) {
            continue;
        }
        
        // 计算相同邻居 - 使用排序和双指针法提高效率
        std::vector<uint32_t> sorted1 = neighbors1;
        std::vector<uint32_t> sorted2 = neighbors2;
        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());
        
        size_t same_count = 0;
        size_t idx1 = 0, idx2 = 0;
        while (idx1 < sorted1.size() && idx2 < sorted2.size()) {
            if (sorted1[idx1] < sorted2[idx2]) {
                idx1++;
            } else if (sorted1[idx1] > sorted2[idx2]) {
                idx2++;
            } else {
                same_count++;
                idx1++;
                idx2++;
            }
        }
        
        total_same_neighbors += same_count;
        total_neighbors_in_graph1 += neighbors1.size();
        nodes_processed++;
        
        if (i % 10000 == 0) {
            std::cout << "Processed " << i << " nodes..." << std::endl;
        }
    }
    
    // 输出详细统计信息
    std::cout << "\n=== Comparison Results ===" << std::endl;
    std::cout << "Graph 1: " << file1 << std::endl;
    std::cout << "Graph 2: " << file2 << std::endl;
    std::cout << "Nodes successfully processed: " << nodes_processed << std::endl;
    
    if (total_neighbors_in_graph1 == 0) {
        std::cout << "Warning: No neighbors found in graph1 for the processed nodes." << std::endl;
        return 0.0;
    }
    
    double ratio = static_cast<double>(total_same_neighbors) / total_neighbors_in_graph1;
    
    std::cout << "Total neighbors in graph1: " << total_neighbors_in_graph1 << std::endl;
    std::cout << "Total same neighbors: " << total_same_neighbors << std::endl;
    std::cout << "Same neighbors ratio: " << ratio << " (" << (ratio * 100) << "%)" << std::endl;
    
    // 额外统计信息
    if (nodes_processed > 0) {
        double avg_neighbors_graph1 = static_cast<double>(total_neighbors_in_graph1) / nodes_processed;
        double avg_same_neighbors = static_cast<double>(total_same_neighbors) / nodes_processed;
        std::cout << "Average neighbors per node in graph1: " << avg_neighbors_graph1 << std::endl;
        std::cout << "Average same neighbors per node: " << avg_same_neighbors << std::endl;
    }
    
    return ratio;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <graph1_path> <graph2_path> [num_nodes]" << std::endl;
    std::cout << "  graph1_path: Path to the first graph file" << std::endl;
    std::cout << "  graph2_path: Path to the second graph file" << std::endl;
    std::cout << "  num_nodes:   Number of nodes to compare (default: 100000)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example: " << program_name << " /path/to/graph1.bin /path/to/graph2.bin 50000" << std::endl;
}

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 获取图文件路径
    std::string graph1_path = argv[1];
    std::string graph2_path = argv[2];
    
    // 设置要比较的节点数量（默认为100000）
    size_t num_nodes_to_compare = 100000;
    if (argc >= 4) {
        try {
            num_nodes_to_compare = std::stoul(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid number of nodes specified. Using default value of 100000." << std::endl;
            std::cerr << "Error details: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Graph Comparison Tool" << std::endl;
    std::cout << "====================" << std::endl;
    
    // 计算相同邻居比例
    double similarity_ratio = calculate_same_neighbors_from_files(
        graph1_path, graph2_path, num_nodes_to_compare);
    
    std::cout << "\nFinal similarity ratio: " << similarity_ratio << std::endl;
    
    return 0;
}