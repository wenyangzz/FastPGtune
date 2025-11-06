#include "dist_list_pool.h"
#include <iostream>

namespace diskann {
    DistList::DistList(int numelements1) {
        numelements = numelements1;
        dists = new dl_type[numelements];
        for (int i = 0; i < numelements; i++) {
            dists[i] = -1;
        }
    }

    void DistList::reset() {
        for (int i = 0; i < record.size(); i++) {
            dists[record[i]] = -1;
        }
        record.resize(0);
    }

    DistList::~DistList() {
        delete[] dists;
    }

    DistListPool::DistListPool(int initmaxpools, int numelements1) : numelements(numelements1) {
        std::cout << "DistListPool init" << std::endl;
        for (int i = 0; i < initmaxpools; i++) {
            dist_pool.push_front(new DistList(numelements));
        }
        std::cout << "DistListPool init" << std::endl;
    }

    int DistListPool::getPoolSize() {
        int size;
        {
            std::unique_lock<std::mutex> lock(poolguard2);
            size = dist_pool.size();
        }
        return size;
    }

    DistList* DistListPool::getFreeVisitedList() {
        DistList* dist_rez;
        {
            std::unique_lock<std::mutex> lock(poolguard2);
            if (dist_pool.size() > 0) {
                dist_rez = dist_pool.front();
                dist_pool.pop_front();
            } else {
                dist_rez = new DistList(numelements);
            }
        }
        dist_rez->reset();
        return dist_rez;
    }

    void DistListPool::releaseDistList(DistList* dl) {
        std::unique_lock<std::mutex> lock(poolguard2);
        dist_pool.push_front(dl);
    }

    DistListPool::~DistListPool() {
        while (dist_pool.size()) {
            DistList* dist_rez = dist_pool.front();
            dist_pool.pop_front();
            delete dist_rez;
        }
    }
}