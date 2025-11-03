#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include<omp.h>

#include <unordered_set>

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
   std:: cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_num * i + j]);  //answer 存储了查询向量精确的近邻结果。
        }
    }
}

static float
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        //auto rs = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> result =  appr_alg.searchKnn(massQ + vecdim * i, k);
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
    

    return 1.0f * correct / total;
}


static float
test_approx2(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,std::vector<size_t> efs) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        //auto rs = appr_alg.searchKnn(massQ + vecdim * i, k);
        auto result =  appr_alg.searchKnn2(massQ + vecdim * i, k,efs);
       // std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        auto gt = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::unordered_set<labeltype> g;    //unordered_set 无序的集合
        total += result.size(); //？是不是用result.size()  
     
        while (gt.size()) {

            g.insert(gt.top().second); //g中存了query[i]的Gt结果
            gt.pop();
        }

        while (result[0].size()) {
            if (g.find(result[0].top().second) != g.end()) {
                correct++;
            } else {
            }
            result[0].pop();
        }

    }
    

    return 1.0f * correct / total;
}
static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    std::vector<size_t> efs;// = { 10,10,10,10,10 };
    // for (int i = k; i < 30; i+=5) {  //填充ef数组，ef>=k
    //     efs.push_back(i);
    // }
    for (int i = 100; i < 500; i += 100) {
        efs.push_back(i);
    }
    // for (int i = 100; i < 500; i += 40) {
    //     efs.push_back(i);
    // }
    //efs.push_back(50);
    for (size_t ef : efs) {
        
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            std::cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }

        StopW stopw2 = StopW();

        recall = test_approx2(massQ, vecsize, qsize, appr_alg, vecdim, answers, k,efs);
        time_us_per_query = stopw2.getElapsedTimeMicro() / qsize;
        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            std::cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}


static void
test_vs_recall2(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    std::vector<size_t> efs;// = { 10,10,10,10,10 };
    for (int i = k; i < 30; i+=5) {  //填充ef数组，ef>=k
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 30) {
        efs.push_back(i);
    }
    // for (int i = 100; i < 500; i += 40) {
    //     efs.push_back(i);
    // }
    for (size_t ef : efs) {
        
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            std::cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}


void sift_test1M() {
	
//后续任务 ： 使用传参的方式确定数据集和参数配置，参考综述。	
	int subset_size_milllions = 1;
	int efConstruction = 40;
	int M = 16;
	

    size_t vecsize = subset_size_milllions * 10000; 

    size_t gt_num = 100;
    size_t qsize = 100; 
    size_t vecdim = 128;
    char path_index[1024];
    char path_gt[1024];
   // char *path_q = "../bigann/bigann_query.bvecs";
   // char *path_data = "../bigann/bigann_base.bvecs"; sift\sift_base.fvecs
    char const *path_q = "../dataset/sift1M/sift_sample_query.fvecs";
    char const *path_data = "../dataset/sift1M/sift_sample_base.fvecs";
    // char *path_q = "../dataset/sift/sift_query.fvecs";
    // char *path_data = "../dataset/sift/sift_base.fvecs";
    std::cout<<"path_q"<<*path_q<<std::endl;
    std::cout<<"path_data"<<*path_data<<std::endl;
    sprintf(path_index, "sift_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);

    //sprintf(path_gt, "../dataset/sift/sift_groundtruth.ivecs");
    sprintf(path_gt, "../dataset/sift1M/sift_sample_groundtruth.ivecs");
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
        if (in != 128) {  
           std::cout << "file error";
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
    HierarchicalNSW<float> *appr_alg ;
    if (exists_test(path_index)) { 
        std::cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);  //HierarchicalNSW函数有多个重载，该处读入老index
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n"; //统计内存占用
    } else {
        std::cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction); //新建index
        //hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&l2space, 2 * vecsize);


    
        input.read((char *) &in, 4); //读入1个原始向量
        if (in != 128) {
            std::cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in * 4);

        for (int j = 0; j < vecdim; j++) {
            mass[j] = massb[j] * (1.0f);
        }

        appr_alg->addPoint((void *) (massb), (size_t) 0); //向索引中加入元素
        int j1 = 0;
        StopW stopw = StopW();   //统计时间
        StopW stopw_full = StopW();
        size_t report_every = 1000; //设为数据集大小的1/10较好
#pragma omp parallel for    //使用多个线程读取原始向量并添加到索引
        for (int i = 1; i < vecsize; i++) {
            //unsigned char mass[128];
            float mass[128];
            int j2=0;
#pragma omp critical //CRITICAL指令指定一块同一时间只能被一条线程执行的代码区域
            {

                input.read((char *) &in, 4);
                if (in != 128) {
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
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (mass), (size_t) j2); //j2是label ,值就是添加节点的顺序值


        }
        input.close();
        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n"; //输出建索引的总时间
        appr_alg->saveIndex(path_index);  //save index
    }

    
    std::vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 1;
    std::cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, gt_num,l2space, vecdim, answers, k);
    std::cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    
    // delete alg_brute;
    // delete appr_alg;

    return;

}

int main() {
    //sift_test1B();
    sift_test1M();
    return 0;
};

