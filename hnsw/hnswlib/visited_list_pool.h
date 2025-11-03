#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
    typedef unsigned short int vl_type; //2Byte
   // typedef float vl_type2;
    class VisitedList {
    public:
        vl_type curV;
        vl_type *mass;
       // vl_type2 *mass2;
        unsigned int numelements;

        VisitedList(int numelements1) {
            curV = -1;
            numelements = numelements1;
            mass = new vl_type[numelements]; //使用mass数组记录所有元素是否被访问 
          //  mass2 = new vl_type2[numelements];
        }

        void reset() {
            curV++;
            if (curV == 0) {
                memset(mass, 0, sizeof(vl_type) * numelements);  //将mass数组元素置0 
                curV++;
            }
           //  memset(mass2, -1, sizeof(vl_type2) * numelements);
        };

        ~VisitedList() { 
            delete[] mass;
            //delete[] mass2;
         }
    };
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

    class VisitedListPool {  
        std::deque<VisitedList *> pool;  //有多个thread，因此有多个VisitedLists ,pool是一个双端队列
        std::mutex poolguard;
        int numelements;

    public:
        VisitedListPool(int initmaxpools, int numelements1) {
            numelements = numelements1;
            for (int i = 0; i < initmaxpools; i++)
                pool.push_front(new VisitedList(numelements));
        }

        VisitedList *getFreeVisitedList() { //从pool中头部取得一个VisitedList（要先复位）
            VisitedList *rez;
            {
                std::unique_lock <std::mutex> lock(poolguard); //上锁
                if (pool.size() > 0) {
                    rez = pool.front(); 
                    pool.pop_front();
                } else {
                    rez = new VisitedList(numelements);
                }
            }
            rez->reset();
            return rez;
        };

        void releaseVisitedList(VisitedList *vl) {
            std::unique_lock <std::mutex> lock(poolguard);
            pool.push_front(vl);
        };

        ~VisitedListPool() {
            while (pool.size()) {
                VisitedList *rez = pool.front();
                pool.pop_front();
                delete rez;
            }
        };
    };
}

