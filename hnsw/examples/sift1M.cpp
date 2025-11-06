// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../hnswlib/hnswlib.h"
#include "../hnswlib/hnswalg.h"
//#include "../hnswlib/hnswalg_advanced.h"
#include <assert.h>
#include<thread>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include <iostream>
#include <unordered_set>
#include<omp.h>

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
using namespace hnswlib;

// namespace
// {

    using idx_t = hnswlib::labeltype;



// get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, size_t gt_num ,L2Space &l2space,
//        size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {

//     (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers); 
//    // DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
//     cout << qsize << "\n";
//     for (int i = 0; i < qsize; i++) {
//         for (int j = 0; j < k; j++) {
//             answers[i].emplace(0.0f, massQA[gt_num * i + j]);  //answer 存储了查询向量精确的近邻结果。
//         }
//     }

 bool exists_test(const std::string &name){
    std::ifstream f(name.c_str());
    return f.good();
    }

// void reset_dist(float *dist_array,int numelements) {
//    // memset(dist_array, -1.0, sizeof(float) * numelements);
//    for(int i=0;i<numelements;i++){
//        dist_array[i]=-1;
//    }
//     // if(n2==5000){
//     //     std::cout<<"numelements == 5000 " <<std::endl;
//     //     std::cout<<dist_array[0] <<std::endl;
//     // }
// }

// void p1(HierarchicalNSW<float> *alg_hnsw2,HierarchicalNSW<float> *alg_hnsw3,DistListPool *dist_list_pool,int n,int s ,std::vector<float> data){
//     int d=128;
//     for (int i = s; i < n; ++i) {
        
//         DistList *dl = dist_list_pool->getFreeVisitedList();
//         dl_type *dist_array = dl->dists;
//       //  #pragma omp ordered
//         if(i==500){
//             std::cout<<"dist_array[0] " <<dist_array[0]<<std::endl;
//             int count=0;
//             for(int j=0;j<i;j++){
//                 if(dist_array[j]!=-1){
//                     count++;
//                 }
//             }
//             std::cout<<"count " <<count<<std::endl;
//             //std::cout<<"poolsize " <<dist_list_pool_->getPoolSize()<<std::endl;
//         }
//         alg_hnsw2->addPoint2(data.data() + d * i, i,dist_array);
//         alg_hnsw3->addPoint2(data.data() + d * i, i,dist_array);
//         if(i==500){
//             std::cout<<"dist_array[0] " <<dist_array[0]<<std::endl;
//             int count=0;
//             for(int j=0;j<i;j++){
//                 if(dist_array[j]!=-1){
//                     count++;
//                 }
//             }
//             std::cout<<"count " <<count<<std::endl;
//             //std::cout<<"poolsize " <<dist_list_pool_->getPoolSize()<<std::endl;
//         }
//         //alg_hnsw4->addPoint2(data.data() + d * i, i,dist_array);
        
//          dist_list_pool->releaseDistList(dl);
//     }
// }
// tableint cur_id(){
//             tableint cur_c = 0;
//             {
//                 // Checking if the element with the same label already exists
//                 // if so, updating it *instead* of creating a new element.
//                 std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
//                 auto search = label_lookup_.find(label);
//                 if (search != label_lookup_.end()) {
//                     tableint existingInternalId = search->second;
//                     templock_curr.unlock();

//                     std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

//                     if (isMarkedDeleted(existingInternalId)) {
//                         unmarkDeletedInternal(existingInternalId);
//                     }
//                     updatePoint(data_point, existingInternalId, 1.0);
                    
//                     return existingInternalId;
//                 }

//                 if (cur_element_count >= max_elements_) {
//                     throw std::runtime_error("The number of elements exceeds the specified limit");
//                 };

//                 cur_c = cur_element_count;
//                 cur_element_count++;
//                 label_lookup_[label] = cur_c;
//             }
// }



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
 
 
static void
test_vs_recall2(float *massQ, size_t vecsize, size_t qsize, MultiHierarchicalNSW<float> &appr_alg, size_t vecdim,
               std::vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k ,tableint enter) {
    std::vector<size_t> efs;// = { 10,10,10,10,10 };
    for (int i = k; i < 30; i+=5) {  //填充ef数组，ef>=k
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10) {
        efs.push_back(i);
    }
    // for (int i = 100; i <= 500; i += 25) {
    //     efs.push_back(i);
    // }
    std::cout << efs.size() << "\t" <<std::endl;
    for (size_t ef : efs) {
        //std::cout << ef << "\t" <<std::endl;
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k,enter);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
       // float time_us_per_query = stopw.getElapsedTimeMicro()/1000000;   
        std::cout << ef << "\t" << recall <<"\t" << time_us_per_query << " s\n";
        // if (recall > 1.0) {
        //     std::cout << recall << "\t" << time_us_per_query << " us\n";
        //     break;
        // }
    }
}
void test() {
    std::mutex dist_array_lock;
    int d = 128;
    idx_t n = 1000;
    idx_t nq = 10;
    size_t k = 1;

    std::vector<int> efConstructions;
     for (int i = 50; i <100; i+=25) {  //填充ef数组
        efConstructions.push_back(i);
    }
    for (int i = 100; i <=220; i+=30) {  //填充ef数组
        efConstructions.push_back(i);
    }

    std::vector<int> M;
     for (int i = 8; i <=24; i+=8) {  //填充ef数组，ef>=k
        M.push_back(i);
    }

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    DistListPool *dist_list_pool = new DistListPool(1, n);


    // float* dist_array = new float[n]; //全局变量数组，存放计算过的距离，锁该怎么上?
    // reset_dist(dist_array,n);
     //std::cout<<"dist_array[0] " <<dist_array[0]<<std::endl;

    char path_index[1024];
    sprintf(path_index, "sift_%dm_ef_%d_M_%d.bin");
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;
#pragma omp parallel for
    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
#pragma omp parallel for
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

         
    std::cout<<"data[0]"<<data[0]<<std::endl;
    std::cout<<"data_size"<<data.size()<<std::endl;
    hnswlib::L2Space space(d);
    
    MultiHierarchicalNSW<float> * alg_hnsw;
    MultiHierarchicalNSW<float> * alg_hnsw2;
    MultiHierarchicalNSW<float> * alg_hnsw3;
    MultiHierarchicalNSW<float> * alg_hnsw4;
    MultiHierarchicalNSW<float> * alg_hnsw5;
    MultiHierarchicalNSW<float> * alg_hnsw6;
    hnswlib::AlgorithmInterface<float>* alg_brute;
    
    if (exists_test(path_index)) { 
        std::cout << "Loading index from " << path_index << ":\n";
        // alg_brute  =  new hnswlib::BruteforceSearch<float>(&space, path_index);
        // alg_hnsw = new hnswlib::MultiHierarchicalNSW<float>(&space, path_index, false);  //HierarchicalNSW函数有多个重载，该处读入老index
       // std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n"; //统计内存占用
    } else {
        std::cout << "Building index:\n";
        //alg_brute  = new BruteforceSearch<float>(&space, n);
        // alg_hnsw = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[0]);
        // alg_hnsw2 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[1]);
        // alg_hnsw3 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[2]);
        // alg_hnsw4 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[3]);
        // alg_hnsw5 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[4]);
        // alg_hnsw6 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[5]);
        }
    // hnswlib::SpaceInterface<float> *s = &space; //测试距离计算函数
    // auto fstdistfunc_ = s->get_dist_func();
    // auto dist_func_param_ = s->get_dist_func_param();
    // float dist = fstdistfunc_(data.data()+d*1, data.data()+d*10, dist_func_param_);
    // std::cout<<"dist1: "<<dist<<std::endl;
    // dist = fstdistfunc_(data.data()+d*1, data.data()+d*1, dist_func_param_);
    // std::cout<<"dist2： "<<dist<<std::endl;

    // HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw4 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw5 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw6 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw7 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw8 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw9 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw10 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw11 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);

// #pragma omp parallel for
//     for (int i = 0; i < n; ++i) {
//         alg_brute->addPoint(data.data() + d * i, i);
//     }
     
    //omp_set_num_threads(16);

    //alg_hnsw->saveIndex(path_index);



    //hnswlib::AlgorithmInterface<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    alg_hnsw = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[0]);
    char * data_level0 = alg_hnsw->data_level0();
    alg_hnsw2 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1],efConstructions[1]);
    alg_hnsw3 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[2]);
    alg_hnsw4 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[3]);
    alg_hnsw5 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[4]);
    alg_hnsw6 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[5]);
    MultiHierarchicalNSW<float> *alg_hnsw7 = new MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[6]);
    MultiHierarchicalNSW<float> *alg_hnsw8 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[7]);
    MultiHierarchicalNSW<float> *alg_hnsw9 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[8]);
    MultiHierarchicalNSW<float> *alg_hnsw10 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , data_level0, M[1], efConstructions[9]);
    // MultiHierarchicalNSW<float> *alg_hnsw11 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[4]);
    // MultiHierarchicalNSW<float> *alg_hnsw12 = new hnswlib::MultiHierarchicalNSW<float>(&space, n , M[1], efConstructions[5]);
    // HierarchicalNSW<float> *alg_hnsw10 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    // HierarchicalNSW<float> *alg_hnsw11 = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
    StopW stopw_create2 = StopW();
    // DistList *dl;
    // dl_type *dist_array;

   // omp_set_num_threads(8); //private(dl,dist_array)
    std::mutex print;
    unsigned long long dist_compute=0;
    unsigned long long dist_reset=0;
    std::cout<<"dist_compute " <<dist_compute<<std::endl;
#pragma omp parallel for reduction(+:dist_compute,dist_reset)//schedule(dynamic)

    for (int i = 0; i < n; ++i) {
        
        DistList *dl = dist_list_pool->getFreeVisitedList();
       
       // dl->reset();
        // if(i==i){
        //        // std::cout<<"dist_array[0] " <<dl->dists[0]<<std::endl;
        //         int count=0;
        //         for(int j=0;j<i;j++){
        //             if(dl->dists[j]<0){
        //                 count++;
        //             //std::cout<<"dl->dists " <<j<<std::endl;
        //             }
        //         }
                
        //         // for(int k=0;k<dl->record.size();k++){
        //         //     std::cout<<"record" <<dl->record[k]<<std::endl;
        //         // }
        //         std::cout<<"count11" <<count<<std::endl;
        //          std::cout<<"count2 " <<dl->record.size()<<std::endl;
        //         //std::cout<<"poolsize " <<dist_list_pool_->getPoolSize()<<std::endl;
        //     }
        //dl_type *dist_array = dl->dists;

      //  #pragma omp ordered
        
         tableint cur_id = alg_hnsw->get_cur_id(i);
        //  if(i%1000==0){
        //     tableint enter2 = alg_hnsw2->get_curr();
        //  tableint enter3 = alg_hnsw3->get_curr();
        //  tableint enter4 = alg_hnsw4->get_curr();
        //   std::cout<<"enter2 " <<enter2<<std::endl;
        //   std::cout<<"enter3 " <<enter3<<std::endl;
        //   std::cout<<"enter4 " <<enter4<<std::endl;
        //  }
         unsigned long long dist_compute2=0;
        dist_compute2 = alg_hnsw->addPoint2(data.data() + d * i, i,-1,dl,cur_id,0);
        dist_compute+=dist_compute2;
        dist_compute2 = alg_hnsw2->addPoint3(data.data() + d * i, i,-1,dl,cur_id,1);
        dist_compute+=dist_compute2;
        dist_compute2 = alg_hnsw3->addPoint3(data.data() + d * i, i,-1,dl,cur_id,1);
        dist_compute+=dist_compute2;
        dist_compute2 = alg_hnsw4->addPoint3(data.data() + d * i, i,-1,dl,cur_id,1);
        dist_compute+=dist_compute2;
        dist_compute2 = alg_hnsw5->addPoint3(data.data() + d * i, i,-1,dl,cur_id,1);
        dist_compute+=dist_compute2;
        dist_compute2 = alg_hnsw6->addPoint3(data.data() + d * i, i,-1,dl,cur_id,1);
        dist_compute+=dist_compute2;
        dist_list_pool->releaseDistList(dl);
        //dist_reset += dl->record.size();
         //std::cout<<"recordsize " <<dl->record.size()<<std::endl;
         
        
        // std::cout<<"recordsize " <<dl->record.size()<<std::endl;
        // print.lock();
        // std::cout<<"dist_array:  "<<&dist_array<<" \ti:  "<<i<<std::endl;
        // //std::cout<<"i:  "<<i<<"thread_num: " <<omp_get_thread_num()<<std::endl;
        // print.unlock();
        // alg_hnsw2->addPoint(data.data() + d * i, i);
        // alg_hnsw3->addPoint(data.data() + d * i, i);

        // if(true){
        //    // std::cout<<"dist_array[0] " <<dl->dists[0]<<std::endl;
        //     int count=0;
        //     for(int j=0;j<i;j++){
        //         if(dl->dists[j]!=-1){
        //             count++;
        //         }
        //     }
        //     std::cout<<"count " <<count<<std::endl;
        //     std::cout<<"recordsize " <<dl->record.size()<<std::endl;
        //     //std::cout<<"poolsize " <<dist_list_pool_->getPoolSize()<<std::endl;
        // }
        // alg_hnsw4->addPoint(data.data() + d * i, i);
        
       // std::cout<<"i:  "<<i<<"thread_num: " <<omp_get_thread_num()<<std::endl;
        //alg_hnsw5->addPoint2(data.data() + d * i, i,dist_array);
        // alg_hnsw6->addPoint2(data.data() + d * i, i,dist_array);
        //  alg_hnsw7->addPoint2(data.data() + d * i, i,dist_array);
        //  alg_hnsw8->addPoint2(data.data() + d * i, i,dist_array);
        // alg_hnsw9->addPoint2(data.data() + d * i, i,dist_array);
        // alg_hnsw10->addPoint2(data.data() + d * i, i,dist_array);
        // alg_hnsw11->addPoint2(data.data() + d * i, i,dist_array);
        // if(i==i){
        //        // std::cout<<"dist_array[0] " <<dl->dists[0]<<std::endl;
        //         int count=0;
        //         for(int j=0;j<i;j++){
        //             if(dl->dists[j]!=-1){
        //                 count++;
        //             //std::cout<<"dl->dists " <<j<<std::endl;
        //             }
        //         }
                
        //         // for(int k=0;k<dl->record.size();k++){
        //         //     std::cout<<"record" <<dl->record[k]<<std::endl;
        //         // }
        //         std::cout<<"count13 " <<count<<std::endl;
        //          std::cout<<"count2 " <<dl->record.size()<<std::endl;
        //         //std::cout<<"poolsize " <<dist_list_pool_->getPoolSize()<<std::endl;
        //     }
        //reset_dist(dist_array,n);
         
    }
    std::cout<<"dist_compute " <<dist_compute<<std::endl;
    //std::cout<<"dist_reset " <<dist_reset<<std::endl;
    float time_cost5 = stopw_create2.getElapsedTimeMicro();
    std::cout << "create time_cost: " << 1e-6 * time_cost5<< std::endl;
    //alg_hnsw->saveIndex(path_index);

    std::cout<<"oringnal create index"<<std::endl; 
    StopW stopw_create = StopW();
// #pragma omp parallel for 
//     for (int i = 0; i < n; ++i) {       
//         alg_hnsw->addPoint(data.data() + d * i, i);
//     }
    
// #pragma omp parallel for 
//     for (int i = 0; i < n; ++i) {       
//         alg_hnsw5->addPoint(data.data() + d * i, i);
//     }

// #pragma omp parallel for 
//     for (int i = 0; i < n; ++i) {       
//         alg_hnsw6->addPoint(data.data() + d * i, i);
//     }
 dist_compute=0;
  std::cout<<"dist_compute " <<dist_compute<<std::endl;
// #pragma omp parallel for reduction(+:dist_compute)
//     for (int i = 0; i < n; ++i) {   
//         unsigned long long dist_compute2 =0;
//         float *datacopy  = data.data() + d * i;
//         dist_compute2 = alg_hnsw7->addPoint(datacopy, i);
//         dist_compute+=dist_compute2;
//         dist_compute2 = alg_hnsw8->addPoint(datacopy, i);
//          dist_compute+= dist_compute2;
//         dist_compute2 = alg_hnsw9->addPoint(datacopy, i);
//         dist_compute+=dist_compute2;
//         dist_compute2 = alg_hnsw10->addPoint(datacopy, i);
//         dist_compute+=dist_compute2;
//         dist_compute2 = alg_hnsw11->addPoint(datacopy, i);
//          dist_compute+= dist_compute2;
//         dist_compute2 = alg_hnsw12->addPoint(datacopy, i);
//         dist_compute+=dist_compute2;
// }
    
     std::cout<<"dist_compute " <<dist_compute<<std::endl;
    time_cost5 = stopw_create.getElapsedTimeMicro();
    std::cout << "create time_cost: " << 1e-6 * time_cost5<< std::endl;
   
    std::vector<size_t> efs;

     for (int i = k; i < 100; i+=10) {  //填充ef数组，ef>=k
        efs.push_back(i);
    }
    // for (int i = 30; i < 100; i += 10) {
    //     efs.push_back(i);
    // }
    // for (int i = 100; i <= 1000; i += 25) {
    //     efs.push_back(i);
    // }
    // std::cout<<"max_ef"<<efs.back()<<std::endl;
    // std::cout<<"efs_size"<<efs.size()<<std::endl;
    // StopW stopw = StopW();   //统计时间
    // // test searchKnnCloserFirst of BruteforceSearch
    // tableint enter = alg_hnsw->get_curr();
    // tableint enter2 = alg_hnsw2->get_curr();

    // size_t correct = 0;
    // size_t total = 0;
    // alg_hnsw->setEf(100);
    // for (size_t j = 0; j < nq; ++j) {
    //     const void* p = query.data() + j * d;
    //     auto gd = alg_brute->searchKnn(p, k,enter);
    //     //auto gd = alg_hnsw->searchKnn(p, k);
    //     //auto res = alg_brute->searchKnnCloserFirst(p, k);

    //     auto result = alg_hnsw->searchKnn(p, k,enter);
    //     //if(j==0){std::cout<<"top_candidates:      "<<result.size()<<std::endl; }
    //     //auto res2 = alg_hnsw->searchKnnCloserFirst(p, k);
        
        
    //     if(gd.size() == result.size()){
    //       // std::cout<<"gd.size"<<gd.size()<<std::endl;
    //        //size_t t = gd.size();
    //        std::unordered_set<size_t> g;
    //        total += gd.size();

    //         while (gd.size()) {
    //         g.insert(gd.top().second);
    //         gd.pop();
    //         }
    //         while (result.size()) {
    //             if (g.find(result.top().second) != g.end()) {
    //                 correct++;
    //             } 
    //             result.pop();
    //         }           
    //     }
    // }
    
    // float time_cost = stopw.getElapsedTimeMicro();
    // std::cout<<"correct:      "<<1.0f * correct / total<<std::endl;   
    // std::cout << "time_cost: " << 1e-6 * time_cost<< std::endl;

    // std::cout<<"改进后的查询"<<std::endl; 
    
    // alg_hnsw2->setEf(100);
    // StopW stopw2 = StopW();
    //  correct = 0;
    //  total = 0;
    
    // for (size_t j = 0; j < nq; ++j) {
    //     const void* p = query.data() + j * d;
    //     auto gd = alg_brute->searchKnn(p, k,enter);
    //     auto result = alg_hnsw2->searchKnn(p, k,enter2);
    //     //std::cout<<"metric_distance_computations"<<alg_hnsw3->metric_distance_computations<<std::endl;     
    //     if(gd.size() == result.size()){
    //       // 
    //        //size_t t = gd.size();
    //        std::unordered_set<size_t> g;
    //        total += gd.size();

    //         while (gd.size()) {
    //         g.insert(gd.top().second);
    //         gd.pop();
    //         }
    //         while (result.size()) {
    //             if (g.find(result.top().second) != g.end()) {
    //                 correct++;
    //             } 
    //             result.pop();
    //         }           
    //     }else{
    //         std::cout<<"gd.size"<<gd.size()<<std::endl;
    //         std::cout<<"rs.size"<<result.size()<<std::endl;
    //         break;
    //     }
    // }
   
    // float time_cost2 = stopw2.getElapsedTimeMicro();
    // std::cout<<"correct:      "<<1.0f * correct / total<<std::endl;   
    // std::cout << "time_cost: " << 1e-6 * time_cost2<< std::endl;

//     StopW stopw2 = StopW();
//      correct = 0;
//      total = 0;
    //   tableint enter = alg_hnsw->get_curr();
    //   tableint enter2 = alg_hnsw2->get_curr();
//      std::cout<<"enter"<<enter<<std::endl; 
//      alg_hnsw->setEf(efs[2]);
// #pragma omp parallel for
//     for (size_t j = 0; j < nq; ++j) {
//         const void* p = query.data() + j * d;
        
        
//        // auto gd = alg_hnsw->searchKnn(p, k,enter);
        
//         //alg_hnsw->setEf(efs[0]);
//         //auto res = alg_brute->searchKnnCloserFirst(p, k);

//         auto result = alg_hnsw->searchKnn2(p, k,efs,enter); 
//         auto gd = result.back();
//         //auto res2 = alg_hnsw->searchKnnCloserFirst(p, k);
        
//         if(j==0){std::cout<<"top_candidates:      "<<result[2].size()<<std::endl; }

//         if(gd.size() == result[2].size()){
//           // std::cout<<"gd.size"<<gd.size()<<std::endl;
//           // size_t t = gd.size();
//            std::unordered_set<size_t> g;
//            total += gd.size();

//             while (gd.size()) {
//             g.insert(gd.top().second);
//             gd.pop();
//             }
//             while (result[2].size()) {
//                 if (g.find(result[2].top().second) != g.end()) {
//                     correct++;
//                 } 
//                 result[2].pop();
//             }
//             //std::cout<<"correct:      "<<1.0f * correct / total<<std::endl;   
//         }
//     }
//     std::cout<<"total      "<<total<<std::endl; 
//     std::cout<<"correct:      "<<1.0f * correct / total<<std::endl; 
//     float time_cost2 = stopw2.getElapsedTimeMicro();
//     std::cout << "time_cost: " << 1e-6 * time_cost2<< std::endl;
   

//     StopW stopw3 = StopW();
// for(size_t ef:efs){
//     alg_hnsw->setEf(ef);
// #pragma omp parallel for
//         for (size_t j = 0; j < nq; ++j) {
//             const void* p = query.data() + j * d;
//             auto gd = alg_hnsw->searchKnn(p, k,enter);            
//         }
        
//     }
//     float time_cost3 = stopw3.getElapsedTimeMicro();
//     std::cout <<"oringnal time_cost: " << 1e-6 * time_cost3<< std::endl;    

//     StopW stopw4 = StopW();
   
// #pragma omp parallel for
//         for (size_t j = 0; j < nq; ++j) {
//             const void* p = query.data() + j * d;
//             auto gd = alg_hnsw->searchKnn2(p, k,efs,enter);       
//         }

//     float time_cost4 = stopw4.getElapsedTimeMicro();
//     std::cout << " improved time_cost: " << 1e-6 * time_cost4<< std::endl;

// StopW stopw6 = StopW();
// alg_hnsw->setEf(efs.back());
// #pragma omp parallel for
//         for (size_t j = 0; j < nq; ++j) {
//             const void* p = query.data() + j * d;
//             auto gd = alg_hnsw->searchKnn(p, k,enter);            
//         }
// float time_cost6 = stopw6.getElapsedTimeMicro();
// std::cout << " oringnal time_cost: " << 1e-6 * time_cost6<< std::endl;  

// StopW stopw7 = StopW();
// alg_hnsw->setEf(efs.back());
// #pragma omp parallel for
//         for (size_t j = 0; j < nq; ++j) {
//             const void* p = query.data() + j * d;
//             auto gd = alg_hnsw2->searchKnn(p, k,enter2);    
                    
//         }
// float time_cost7 = stopw7.getElapsedTimeMicro();
// std::cout << " improved time_cost: " << 1e-6 * time_cost7<< std::endl;  

//     //delete alg_brute;
//     delete alg_hnsw;
   
}

//} // namespace
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


// inline bool exists_test(const std::string &name) {
//     std::ifstream f(name.c_str());
//     return f.good();
// }


void sift_test1M() {
	
//后续任务 ： 使用传参的方式确定数据集和参数配置，参考综述。	
	int subset_size_milllions = 1;
	int efConstruction = 150;
	//int M = 16;
	

    size_t vecsize = subset_size_milllions * 1000000; //glove1M是 1183514

    size_t gt_num = 100; 
    size_t qsize = 10000; //gist 1000
    size_t vecdim = 128; //glove是100 128
    char path_index[1024];
    char path_gt[1024];
   // char *path_q = "../bigann/bigann_query.bvecs";
   // char *path_data = "../bigann/bigann_base.bvecs"; sift\sift_base.fvecs
    // char const *path_q = "../dataset/sift1M/sift_sample_query.fvecs";
    // char const *path_data = "../dataset/sift1M/sift_sample_base.fvecs";//dataset\glove1M\glove-100_query.fvecs
     char const *path_q = "../dataset/sift/sift_query.fvecs";
     char const *path_data = "../dataset/sift/sift_base.fvecs";
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
    sprintf(path_gt, "../dataset/sift/sift_groundtruth.ivecs");  //dataset\gist\gist_base.fvecs
    //sprintf(path_gt, "./dataset/gist/gist_groundtruth.ivecs");

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
     std::vector<int> efConstructions;
     std::vector<int> M;
     for (int i = 100; i <150; i+=5) {  //填充ef数组
        efConstructions.push_back(i);
    }
    for (int i = 8; i <=26; i+=2) {  //填充ef数组
        M.push_back(i);
    }
    MultiHierarchicalNSW<float> *appr_alg = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[4], efConstructions[0]);
    char *data_level0 = appr_alg->data_level0();
    MultiHierarchicalNSW<float> *appr_alg2 = new MultiHierarchicalNSW<float>(&l2space, vecsize,data_level0, M[4], efConstructions[1]);
    MultiHierarchicalNSW<float> *appr_alg3 = new MultiHierarchicalNSW<float>(&l2space, vecsize,data_level0, M[4], efConstructions[2]);
    MultiHierarchicalNSW<float> *appr_alg4 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0, M[4], efConstructions[3]);
    MultiHierarchicalNSW<float> *appr_alg5 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0, M[4], efConstructions[4]);
    MultiHierarchicalNSW<float> *appr_alg6 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0,M[4], efConstructions[5]);
    MultiHierarchicalNSW<float> *appr_alg7 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0, M[4], efConstructions[6]);
    MultiHierarchicalNSW<float> *appr_alg8 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0,M[4], efConstructions[7]);
    MultiHierarchicalNSW<float> *appr_alg9 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0,M[4], efConstructions[8]);
    MultiHierarchicalNSW<float> *appr_alg10 = new MultiHierarchicalNSW<float>(&l2space, vecsize, data_level0,M[4], efConstructions[9]);

    // MultiHierarchicalNSW<float> *appr_alg = new MultiHierarchicalNSW<float>(&l2space, vecsize, 16, 200);
    // char *data_level0 = appr_alg->data_level0();
    // MultiHierarchicalNSW<float> *appr_alg2 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[1], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg3 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[2], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg4 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[3], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg5 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[4], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg6 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[5], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg7 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[6], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg8 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[7], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg9 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[8], efConstruction);
    // MultiHierarchicalNSW<float> *appr_alg10 = new MultiHierarchicalNSW<float>(&l2space, vecsize, M[9], efConstruction);
    DistListPool *dist_list_pool = new DistListPool(2, vecsize);

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
        appr_alg->addPoint((void *) (massb), (size_t) 0); //向索引中加入元素
        //appr_alg->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,0);
        // appr_alg->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,0);
        appr_alg2->addPoint((void *) (mass), (size_t) 0);
        appr_alg3->addPoint((void *) (mass), (size_t) 0);
        appr_alg4->addPoint((void *) (massb), (size_t) 0); 
        appr_alg5->addPoint((void *) (mass), (size_t) 0);
        appr_alg6->addPoint((void *) (mass), (size_t) 0);
        appr_alg7->addPoint((void *) (mass), (size_t) 0);
        appr_alg8->addPoint((void *) (mass), (size_t) 0);
        appr_alg9->addPoint((void *) (mass), (size_t) 0);
        appr_alg10->addPoint((void *) (mass), (size_t) 0);
        int j1 = -1;
        StopW stopw = StopW();   //统计时间
        StopW stopw_full = StopW();
        size_t report_every = 100000; //设为数据集大小的1/10较好
        std::cout << "load data";
        unsigned long long dist_compute=0;
        unsigned long long dist_reset=0;
        std::cout<<"dist_compute " <<dist_compute<<std::endl;
// omp_set_num_threads(8);
#pragma omp parallel for reduction(+:dist_compute,dist_reset)   //使用多个线程读取原始向量并添加到索引
        for (int i = 1; i < vecsize; i++) {
            unsigned long long dist_compute2 =0;
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
                    stopw.reset();
                }
            }
            DistList *dl = dist_list_pool->getFreeVisitedList();
            tableint cur_id = appr_alg->get_cur_id(i);
            if(j2%100000==0){
            std::cout<<"dl->dists " <<dl->record.size()<<std::endl;
            //  std::cout<<"dl->dists pair" <<dl->record_pair.size()<<std::endl;
             std::cout<<"dl->dists pair" <<dl->dists_pair_map->size()<<std::endl;
            }
            dist_compute2 = appr_alg->addPoint2((void *) (mass), (size_t) j2,-1,dl,cur_id,0);
            dist_compute+=dist_compute2; 
            dist_compute2 = appr_alg2->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg3->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg4->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg5->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg6->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg7->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg8->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg9->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            dist_compute2 = appr_alg10->addPoint3((void *) (mass), (size_t) j2,-1,dl,cur_id,1);
            dist_compute+=dist_compute2;
            if(j2%100000==0){
            std::cout<<"dl->dists " <<dl->record.size()<<std::endl;
            //  std::cout<<"dl->dists pair" <<dl->record_pair.size()<<std::endl;
             std::cout<<"dl->dists pair" <<dl->dists_pair_map->size()<<std::endl;
            }
        
             dist_list_pool->releaseDistList(dl);

            // dist_compute2 = appr_alg->addPoint((void *) (mass), (size_t) j2); 
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg2->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg3->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg4->addPoint((void *) (mass), (size_t) j2); 
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg5->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg6->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg7->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg8->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg9->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            // dist_compute2 = appr_alg10->addPoint((void *) (mass), (size_t) j2);
            // dist_compute+=dist_compute2;
            
        }
        input.close();
        std::cout<<"dist_compute " <<dist_compute<<std::endl;
        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n"; //输出建索引的总时间
       

    std::vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 1;
    std::cout << "Parsing gt:\n";
     get_gt(massQA, massQ, mass, vecsize, qsize, gt_num,l2space, vecdim, answers, k);
    // std::cout << "Loaded gt\n";
     tableint enter= appr_alg->get_curr();
    for (int i = 0; i < 1; i++)
        test_vs_recall2(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k,enter);
    //     //test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k,enter);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    
    // delete alg_brute;
    // delete appr_alg;

    return;

}

int main() {
    // std::cout << "Testing ..." << std::endl;
    // test();
    // std::cout << "Test ok" << std::endl;


    sift_test1M();
    return 0;
}
