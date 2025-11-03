#pragma once

#include <mutex>
#include <string.h>
#include <deque>
#include <set>
#include <iostream>
#include <map>
#include <unordered_map>
// #include <string> 
namespace hnswlib {
   // typedef unsigned short int vl_type; //2Byte

   struct pair_hash
    {
        template<class T1, class T2>
        std::size_t operator() (const std::pair<T1, T2>& p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

    typedef float dl_type;
    class DistList {
    public:
        //vl_type curV;
        dl_type *dists;
        // dl_type *dists_pair;
        int cnt = 0;
        std::vector<unsigned int> record;
        //std::vector<unsigned int> record_pair;
        
        //std::map<std::pair<tableint,tableint>,float> *dists_pair_map;
        //std::unordered_map<std::pair<tableint,tableint>,float,pair_hash> *dists_pair_map;
        
        //std::unordered_map<std::string,float> *dists_pair_map;
        //std::priority_queue <unsigned int,std::vector<unsigned int>,std::less<unsigned int> > record;
        //std::set<unsigned int> record;
       // vl_type2 *mass2;
        unsigned int numelements;
        // unsigned int numelements2;
        DistList(int numelements1) {
            
            numelements = numelements1;
            dists = new dl_type[numelements]; //使用dists数组记录所有元素是否被访问 
            for(int i=0;i<numelements;i++){
                dists[i]=-1;
             }
            //  numelements2 = 2 * numelements1+100;
            // dists_pair = new dl_type[numelements2]; 
            //  for(int i=0;i<numelements2;i++){
            //     dists_pair[i]=0;
            //  }
             std::cout<<"distcreate "<<std::endl;
             record.reserve(10000);
            // record_pair.reserve(10000);
            //dists_pair_map = new std::unordered_map<std::string,float> ;
            // dists_pair_map = new std::unordered_map<std::pair<tableint,tableint>,float,pair_hash> ;
        }
        // void init_dists_map(){
        //     if(cnt>10)
        //     dists_pair_map = new std::map<std::pair<tableint,tableint>,float>;
        // }
        
        void reset() {
           
             for(int i=0;i<record.size();i++){
                 dists[record[i]]=-1;
             }
              record.resize(0);
           
            // while(!record.empty()){
            //     dists[record.top()]=-1;
            //     record.pop();
            // }
            //std::cout<<"dists_pair_map size "<<record_pair.size()<<std::endl;

            // for(int i=0;i<record_pair.size();i++){
            //      dists_pair[record_pair[i]]=0;
            //  }
            //   record_pair.resize(0);

            // cnt++;
            // if(cnt>0){
            //     //std::cout<<"dists_pair_map size "<<dists_pair_map->size()<<std::endl;
            //      cnt=0;
            //     // delete dists_pair_map;
            //     // dists_pair_map = new std::unordered_map<std::pair<tableint,tableint>,float,pair_hash>;
            //     dists_pair_map->clear();
            // }
        }

        ~DistList() { 
            delete[] dists;
            //delete[] dists_pair;
            //delete record;
            // if(dists_pair_map!=nullptr){
            //     delete dists_pair_map;
            //     dists_pair_map =nullptr;
            // }
            
         }
    };
    //M=16 1 
    // class DistMap{
    //     int cnt = 0;
    //     std::map<std::pair<tableint,tableint>,float> *dists_pair_map;

    //     DistMap(){
    //         dists_pair_map= new std::map<std::pair<tableint,tableint>,float>;
    //     }

    //      void init_dists_map(){
    //         if(cnt>10){
    //             cnt=0;
    //             delete dists_pair_map;
    //             dists_pair_map = new std::map<std::pair<tableint,tableint>,float>;
    //         }
    //     }
    // };
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of distLists
//
/////////////////////////////////////////////////////////

    class DistListPool {  
        std::deque<DistList *> dist_pool;  //有多个thread，因此有多个VisitedLists ,pool是一个双端队列
        std::mutex poolguard2;
        int numelements;

    public:
        DistListPool(int initmaxpools, int numelements1) {
            numelements = numelements1;
            std::cout<<"DistListPool init"<<std::endl;
            for (int i = 0; i < initmaxpools; i++)
                dist_pool.push_front(new DistList(numelements));
            std::cout<<"DistListPool init"<<std::endl;
        }

        int getPoolSize(){
            int size;
            {
                std::unique_lock <std::mutex> lock(poolguard2); //上锁
                size =  dist_pool.size();
            }
            return size;
        }

        DistList *getFreeVisitedList() { //从pool中头部取得一个VisitedList（要先复位）
            DistList *dist_rez;
            {
                std::unique_lock <std::mutex> lock(poolguard2); //上锁
                if (dist_pool.size() > 0) {
                    dist_rez = dist_pool.front(); 
                    dist_pool.pop_front();
                } else {
                    dist_rez = new DistList(numelements);
                }
            }
            dist_rez->reset();
            
            return dist_rez;
        };

        void releaseDistList(DistList *dl) {
            std::unique_lock <std::mutex> lock(poolguard2);
            dist_pool.push_front(dl);
        };

        ~DistListPool() {
            while (dist_pool.size()) {
                DistList *dist_rez = dist_pool.front();
                dist_pool.pop_front();
                delete dist_rez;
            }
        };
    };
}

