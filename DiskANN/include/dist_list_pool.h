#pragma once

#include <mutex>
#include <deque>
#include <vector>
#include <unordered_map>

namespace diskann {
    struct pair_hash {
        template<class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

    typedef float dl_type;

    class DistList {
    public:
        dl_type* dists;
        std::vector<unsigned int> record;
        unsigned int numelements;

        DistList(int numelements1);
        void reset();
        ~DistList();
    };

    class DistListPool {
    private:
        std::deque<DistList*> dist_pool;
        std::mutex poolguard2;
        int numelements;

    public:
        DistListPool(int initmaxpools, int numelements1);
        int getPoolSize();
        DistList* getFreeVisitedList();
        void releaseDistList(DistList* dl);
        ~DistListPool();
    };
}