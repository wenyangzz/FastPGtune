#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include<omp.h>

#include "hnswlib/hnswalg.h"


//using namespace std;
using namespace hnswlib;

// class StopW {
//     std::chrono::steady_clock::time_point time_begin;
// public:
//     StopW() {
//         time_begin = std::chrono::steady_clock::now();
//     }

//     float getElapsedTimeMicro() {
//         std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
//         return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
//     }

//     void reset() {
//         time_begin = std::chrono::steady_clock::now();
//     }

// };



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


//massQA:groundtruth ; massQ query ; mass:original data ;anwser : 是一个vector，每个元素是一个优先队列，每一个优先队列存了一个query的查询结果,查询结果按距离升序排序
static void
get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, size_t gt_num ,L2Space &l2space,
       size_t vecdim,std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {

    (std::vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers); 
   // DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
   
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_num * i + j]);  //answer 存储了查询向量精确的近邻结果。
        }
    }
    std:: cout << "get_gt success" << "\n";
}

static float
test_approx(float *massQ, size_t vecsize, size_t qsize, MultiHierarchicalNSW<float> &appr_alg, size_t vecdim,
            std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,tableint enter) {
    size_t correct = 0;
    size_t total = 0;
    // tableint enter = enter;
    //std::cout<<"enter"<<enter<<std::endl; 
    //uncomment to test in parallel mode:
    #pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        //auto rs = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> result =  appr_alg.searchKnn(massQ + vecdim * i, k,enter);
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        std::unordered_set<labeltype> g;    //unordered_set 无序的集合
        total += result.size(); //？是不是用result.size() 
       
        while (gt.size()) {

            g.insert(gt.top().second); //g中存了query[i]的Gt结果
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
    }
    
    return  1.0f * correct/total ;
}


static float
test_approx2(float *massQ, size_t vecsize, size_t qsize, MultiHierarchicalNSW<float> &appr_alg, size_t vecdim,
            std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,std::vector<size_t> efs,tableint enter) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    // tableint enter = enter;
    std::cout<<"enter"<<enter<<std::endl; 
    StopW stopw3 = StopW();
    for(size_t ef:efs){ //原查询算法总时间
    appr_alg.setEf(ef);
    //qsize=10000;
    #pragma omp parallel for 
    for (size_t j = 0; j < qsize; ++j) {    
            auto gd = appr_alg.searchKnn(massQ + vecdim * j, k,enter);       
        }  
    }
    float time_cost3 = stopw3.getElapsedTimeMicro();
    std::cout <<"oringal search method\t"<< "time_cost:\t " << 1e-6 * time_cost3<< std::endl; 
    
    StopW stopw4 = StopW();
    #pragma omp parallel for       //新算法查询时间
        for (size_t j = 0; j < qsize; ++j) {  
            auto gd = appr_alg.searchKnn2(massQ + vecdim * j, k,efs,enter);       
        }
        float time_cost4 = stopw4.getElapsedTimeMicro();
    std::cout<<"Improved search method\t" << "time_cost:\t " << 1e-6 * time_cost4<< std::endl;    


    appr_alg.setEf(efs.back()); //原查询算法只执行max ef时查询时间
    StopW stopw1 = StopW();
    #pragma omp parallel for
        for (size_t j = 0; j < qsize; ++j) {
            auto gd = appr_alg.searchKnn(massQ + vecdim * j, k,enter);       
        }        
    float time_cost1 = stopw1.getElapsedTimeMicro();
    std::cout <<"efs"<<efs.back()<< "\ttime_cost:\t " << 1e-6 * time_cost1<< std::endl;  

     //return 1.0f * correct / total;
     return 1.0f * correct ;
}


static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, MultiHierarchicalNSW<float> &appr_alg, size_t vecdim,
               std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,tableint enter) {
    std::cout<<"test_vs_recall"<<std::endl;
    std::vector<size_t> efs;// = { 10,10,10,10,10 };
    for (int i = k; i < 30; i+=5) {  //填充ef数组，ef>=k
        efs.push_back(i);
    }
    for (int i = 1000; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i <= 1000; i += 25) {
        efs.push_back(i); 
    }
   // efs.push_back(20);
    std::sort(efs.begin(),efs.end());
    std::cout<<"efsearch nums"<<efs.size()<<std::endl;
    std::cout<<"efs[0]"<<efs[0]<<std::endl;
    StopW stopw2 = StopW();

    float recall = test_approx2(massQ, vecsize, qsize, appr_alg, vecdim, answers, k,efs,enter);
    float time_us_per_query = stopw2.getElapsedTimeMicro() / qsize;
    std::cout <<"新旧查询一致率\t" << recall << std::endl;
               
}


static void
test_vs_recall2(float *massQ, size_t vecsize, size_t qsize, MultiHierarchicalNSW<float> &appr_alg, size_t vecdim,
               std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k ,tableint enter,
                int efQuery,             // 新增：查询时使用的 ef
                std::ostream &out        ) {
    // std::vector<size_t> efs;// = { 10,10,10,10,10 };
    // for (int i = k; i < 30; i+=5) {  //填充ef数组，ef>=k
    //     efs.push_back(i);
    // }
    // for (int i = 30; i < 100; i += 10) {
    //     efs.push_back(i);
    // }

    const int repeat = 5;  // 重复次数
    // for (int i = 10; i < 50; i += 5) {
    //     efs.push_back(i);
    // }
    // for (int i = 50; i < 200; i += 25) {
    //     efs.push_back(i);
    // }
    // for (int i = 200; i < 400; i += 50) {
    //     efs.push_back(i);
    // }
    // for (int i = 400; i <= 1000; i += 100) {
    //     efs.push_back(i);
    // }
    // std::cout << efs.size() << "\t" <<std::endl;
    // for (size_t ef : efs) {
        std::cout << efQuery << "\t" <<std::endl;
        float sum_recall = 0.0;
        float sum_qps    = 0.0;
        for(int r = 0;r < repeat;r++)
        {
            appr_alg.setEf(efQuery);
            StopW stopw = StopW();

            float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k,enter);
            float qps = (float)qsize * 1e6 / stopw.getElapsedTimeMicro();
            sum_recall += recall;
            sum_qps    += qps; 
        }
        // appr_alg.setEf(ef);
        // StopW stopw = StopW();
        float avg_recall = sum_recall / repeat;
        float avg_qps    = sum_qps    / repeat;
        // float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k,enter);
        // float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
       // float time_us_per_query = stopw.getElapsedTimeMicro()/1000000;
        out << efQuery << ","
            << avg_recall << ","
            << avg_qps << "\n";   
        std::cout << efQuery << "\t" << avg_recall <<"\t" << avg_qps << " s\n";
        // if (recall > 1.0) {
        //     std::cout << recall << "\t" << time_us_per_query << " us\n";
        //     break;
        // }
    // }
}

inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}


void sift_test1M(int argc, char **argv) {
	//传入参数查看
     if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " n M1 efBuild1 efQuery1 out1 "
                     "M2 efBuild2 efQuery2 out2 … Mn efBuildn efQueryn outn\n";
        return;
    }
    int n = std::stoi(argv[1]);
    if (argc != 4 * n + 2) {
        std::cerr << "Error: Expecting exactly 4*n parameters after n." << std::endl;
        return;
    }
    std::vector<int>    M_vals(n), ef_vals(n), efQuery_vals(n);
    std::vector<std::string> out_files(n);
    for (int i = 0; i < n; ++i) {
        int base = 2 + 4 * i;
        M_vals[i]       = std::stoi(argv[base + 0]);
        ef_vals[i] = std::stoi(argv[base + 1]);
        efQuery_vals[i] = std::stoi(argv[base + 2]);
        out_files[i]    = argv[base + 3];
    }



//后续任务 ： 使用传参的方式确定数据集和参数配置，参考综述。	
	int subset_size_milllions = 1;
	// int efConstruction = 200;
	//int M = 16;
	
    size_t vecsize = subset_size_milllions * 1000000; //glove1M是 1183514，msong是992272

    size_t gt_num = 100; //msong是20
    size_t qsize = 10000; //gist 1000；sift1M是10000；glove 10000;msong 是 200
    size_t vecdim = 128; //glove是100 128 gist是960,msong是420
    char path_index[1024];
    char path_gt[1024];
     char const *path_q = "/mnt/data2/zwy/zhou/ATHCP/dataset/sift/sift_query.fvecs";
     char const *path_data = "/mnt/data2/zwy/zhou/ATHCP/dataset/sift/sift_base.fvecs";
    //  char const *path_q = "/mnt/data2/zhou/dataset/glove1M/glove-100_query.fvecs";
    //  char const *path_data = "/mnt/data2/zhou/dataset/glove1M/glove-100_base.fvecs";
        // char const *path_q = "/mnt/data2/zwy/zhou/ATHCP/dataset/glove1M/glove-100_query.fvecs";
        // char const *path_data = "/mnt/data2/zwy/zhou/ATHCP/dataset/glove1M/glove-100_base.fvecs";
    // "D:\\Projects\\VScodeProjects\\dataset\\sift1M"
    // char const *path_q = "/mnt/data2/zhou/dataset/msong/msong_query.fvecs";
    //  char const *path_data = "/mnt/data2/zhou/dataset/msong/msong_base.fvecs";
    // char const *path_q = "./dataset/glove1M/glove-100_query.fvecs";
    // char const *path_data = "./dataset/glove1M/glove-100_base.fvecs";
    //  char const *path_q = "./dataset/gist/gist_query.fvecs";
    //  char const *path_data = "./dataset/gist/gist_base.fvecs";
    
    std::cout<<"path_q"<<*path_q<<std::endl;
    std::cout<<"path_data"<<*path_data<<std::endl;
    //sprintf(path_index, "gist_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);
    //sprintf(path_index, "glove_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);

    //sprintf(path_gt, "./dataset/glove1M/glove-100_groundtruth.ivecs"); 
    //sprintf(path_gt, "../dataset/sift1M/sift_sample_groundtruth.ivecs"); dataset\sift\sift_groundtruth.ivecs
    //sprintf(path_gt, "D:\\Projects\\VScodeProjects\\dataset\\sift1M\\sift_groundtruth.ivecs");  //dataset\gist\gist_base.fvecs
    //sprintf(path_gt, "./dataset/gist/gist_groundtruth.ivecs");
    sprintf(path_gt, "/mnt/data2/zwy/zhou/ATHCP/dataset/sift/sift_groundtruth.ivecs");
    // sprintf(path_gt, "/mnt/data2/zhou/dataset/msong/msong_groundtruth.ivecs");
    // sprintf(path_gt, "/mnt/data2/zhou/dataset/glove1M/glove-100_groundtruth.ivecs"); 
    
    std::cout<<"path_gt"<<path_gt<<std::endl;
    std::cout<<"path_q"<<*path_q<<std::endl;
    std::cout<<"path_data"<<*path_data<<std::endl;
//sift_sample_groundtruth.ivecs
    //unsigned char *massb = new unsigned char[vecdim];
    float *massb = new float[vecdim];
    std::cout << "Loading GoundTruth:\n";
    std::ifstream inputGT(path_gt, std::ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 100]; //groundturth 存储的是精确结果的邻居的id
    for (int i = 0; i < qsize; i++) { 
        int t; 
        inputGT.read((char *) &t, 4); 
        inputGT.read((char *) (massQA + 100 * i), t * 4); 
        if (t != 100) {   
            std::cout << "err"; 
            return; 
        } 
    } 
    inputGT.close(); 
	 
    std::cout << "Loading queries:\n"; 
    //unsigned char *massQ = new unsigned char[qsize * vecdim]; //unsigned char size = 1 byte bvec中向量的每一维分量用1字节存储 //存储查询向量的一维数组
    float *massQ = new float[qsize * vecdim]; 
    std::ifstream inputQ(path_q, std::ios::binary); 
 
    for (int i = 0; i < qsize; i++) { 
        int in = 0; 
        inputQ.read((char *) &in, 4); //向量的首4个字节存储的是该向量的维度 对sift是128 
        if (in != vecdim) {  //glove 100 
           std::cout << "file error";
           std::cout << in;
            exit(1);
        }
        inputQ.read((char *) massb, in * 4); //读入一整个向量 也就是128*4个字节，
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }

    }
    inputQ.close();
    std::cout << "load query successfully";

    //unsigned char *mass = new unsigned char[vecdim]; //存储原始向量的一维数组
    float *mass = new float[vecdim];
    std::ifstream input(path_data, std::ios::binary);
    int in = 0;
    L2Space l2space(vecdim); //L2Space 用于fvec 数据类型float ;L2SpaceI用于bvec 数据类型unsigned char

    //HierarchicalNSW<int> *appr_alg; //判断是否存在Hnsw索引，若无则新建索引
    //  std::vector<int> efConstructions;
    //  std::vector<int> M;
    //  for (int i = 100; i <150; i+=5) {  //填充ef数组
    //     efConstructions.push_back(i);
    // }
    // for (int i = 8; i <=26; i+=2) {  //填充ef数组
    //     M.push_back(i);
    // }
    std::vector<MultiHierarchicalNSW<float>*> appr_algs(n);
    for (int i = 0; i < n; ++i) {
        appr_algs[i] = new MultiHierarchicalNSW<float>(&l2space, vecsize, M_vals[i], ef_vals[i]);
    }
    // MultiHierarchicalNSW<float> *appr_alg = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[4], efConstructions[0]);
    // MultiHierarchicalNSW<float> *appr_alg2 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[4], efConstructions[1]);
    // MultiHierarchicalNSW<float> *appr_alg3 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[4], efConstructions[2]);
    DistListPool *dist_list_pool = new DistListPool(70, vecsize);

    // if (exists_test(path_index)) { 
    //     std::cout << "Loading index from " << path_index << ":\n";
    //     appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);  //HierarchicalNSW函数有多个重载，该处读入老index
    //     std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n"; //统计内存占用
    // } else {
        std::cout << "Building index:\n";


        std::cout << "load data";
        input.read((char *) &in, 4); //读入1个原始向量
        if (in != vecdim) { //glove 100
            std::cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in * 4);

        for (int j = 0; j < vecdim; j++) {
            mass[j] = massb[j] * (1.0f);
        }
        std::cout << "load data";
        // appr_alg->addPoint((void *) (massb), (size_t) 0); //向索引中加入第一个元素
        // appr_alg2->addPoint((void *) (mass), (size_t) 0);
        // appr_alg3->addPoint((void *) (mass), (size_t) 0);
        for (int j = 0; j < n; ++j) {
            appr_algs[j]->addPoint((void *) (mass), (size_t) 0);
        }
        int j1 = 0;
        StopW stopw = StopW();   //统计时间
        StopW stopw_full = StopW();
        size_t report_every = 100000; //设为数据集大小的1/10较好
        std::cout << "load data";
        unsigned long long dist_compute=0;
        unsigned long long dist_reset=0;
        
        std::cout<<"dist_compute " <<dist_compute<<std::endl;
        //omp_set_num_threads(8);
        
        std::cout<<"omp_get_max_threads() " <<omp_get_max_threads()<<std::endl;
        const std::string log_file4 = 
        "/home/yinzh/hnsw/参数实验/memory.log";
        std::ofstream ofs4(log_file4, std::ios::out | std::ios::app);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
        if (!ofs4) {
                    std::cerr << "无法打开日志文件: " << log_file4 << std::endl;
                    } else {
                        ofs4<< "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs2 << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs.close(); // ofstream 在析构时会自动关闭
                    }     
#pragma omp parallel for reduction(+:dist_compute,dist_reset)   //使用多个线程读取原始向量并添加到索引
        for (int i = 1; i < vecsize; i++) {
            // unsigned long long dist_compute2 =0;
            //unsigned char mass[128];
            float *mass = new float[vecdim]; 
            int j2=0;
#pragma omp critical //CRITICAL指令指定一块同一时间只能被一条线程执行的代码区域
            {

                input.read((char *) &in, 4);
                if (in != vecdim) { //glove 100
                    std::cout << "file error";
                    exit(1);
                }
                input.read((char *) massb, in * 4);
                for (int j = 0; j < vecdim; j++) {
                    mass[j] = massb[j];
                }
                j1++;
                j2=j1;
                if (j1 % report_every == 0) { //每添加N条向量，打印一次时间和内存信息
                    std::cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";

                    if (!ofs4) {
                    std::cerr << "无法打开日志文件: " << log_file4 << std::endl;
                    } else {
                        ofs4 << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs2 << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs.close(); // ofstream 在析构时会自动关闭
                    }     
                    stopw.reset();
                }
            }
            
            DistList *dl = dist_list_pool->getFreeVisitedList();
            tableint cur_id = appr_algs[0]->get_cur_id(i);//
            // dist_compute2 = appr_alg->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,0);
            // dist_compute+=dist_compute2; 
            // dist_compute2 = appr_alg2->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            // dist_compute+=dist_compute2;
            for (int j = 0; j < n; ++j) {
                auto cmps = appr_algs[j]->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,0);
                dist_compute+=cmps.first;
                dist_reset+=cmps.second;
                // unsigned long long cmp = 0;
                // appr_algs[j]->addPoint((void *) (mass), (size_t) j2,-1,cmp);
                // dist_compute+=cmp;
            }
            dist_list_pool->releaseDistList(dl); 
        }
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
        if (!ofs4) {
                    std::cerr << "无法打开日志文件: " << log_file4 << std::endl;
                    } else {
                        ofs4<< "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs2 << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                        // ofs.close(); // ofstream 在析构时会自动关闭
                    }     
        input.close();
        const std::string log_file = 
        "/home/yinzh/hnsw/参数实验/computer_dist1.log";
        const std::string log_file2 = 
        "/home/yinzh/hnsw/参数实验/computer_prune1.log";
        const std::string log_file1 = 
        "/home/yinzh/hnsw/参数实验/index_time1.log";
        std::ofstream ofs1(log_file, std::ios::out | std::ios::app);
        if (!ofs1) {
                std::cerr << "无法打开日志文件: " << log_file << std::endl;
            } else {
                ofs1 << "dist_compute: " << dist_compute << "\n";
                // ofs.close(); // ofstream 在析构时会自动关闭
            }
         std::ofstream ofs3(log_file2, std::ios::out | std::ios::app);
        if (!ofs3) {
                std::cerr << "无法打开日志文件: " << log_file2 << std::endl;
            } else {
                ofs3 << "computer_prune: " << dist_reset << "\n";
                // ofs.close(); // ofstream 在析构时会自动关闭
            }    
        // std::cout << "dist_compute:  " << dist_compute<< "\n";
        std::ofstream ofs2(log_file1, std::ios::out | std::ios::app);
        if (!ofs2) {
                std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
            } else {
                ofs2 << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
                // ofs2 << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                // ofs.close(); // ofstream 在析构时会自动关闭
            }
        std::cout<<"dist_compute " <<dist_compute<<std::endl;
        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n"; //输出建索引的总时间

    std::vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 10;
    std::cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, gt_num,l2space, vecdim, answers, k);
    std::cout << "Loaded gt\n";
    // tableint enter= appr_algs[1]->get_curr();
    StopW stopw_search = StopW();
    for (int i = 0; i < n; i++)
    { 
        int efQuery = efQuery_vals[i];
        // const std::string &fname = out_files[i];
        std::string csv_file = out_files[i] + "_performance.csv";
        std::ofstream ofs(csv_file, std::ios::out | std::ios::app);
        // std::ofstream out("results.csv", std::ios::app);  // 以追加模式打开文件
        tableint enter= appr_algs[i]->get_curr();
        test_vs_recall2(massQ, vecsize, qsize, *appr_algs[i], vecdim, answers, k,enter,
                         efQuery,   
                            ofs        );
    }
     if (!ofs2) {
                std::cerr << "无法打开日志文件: " << log_file1 << std::endl;
            } else {
                ofs2 << "Search time:" << 1e-6 * stopw_search.getElapsedTimeMicro() << "  seconds\n";
                ofs2 << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
                // ofs.close(); // ofstream 在析构时会自动关闭
            }
        // test_vs_recall2(massQ, vecsize, qsize, *appr_algs[1], vecdim, answers, k,enter);
     std::cout << "Search time:" << 1e-6 * stopw_search.getElapsedTimeMicro() << "  seconds\n"; //输出建索引的总时间
        //test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k,enter);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    
    // delete alg_brute;
    // delete appr_alg;

    return;

}

// int main() {
//     //sift_test1B();
//     sift_test1M();
//     return 0;
// };

