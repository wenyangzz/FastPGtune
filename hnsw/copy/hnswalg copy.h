#pragma once

#include "visited_list_pool.h"
#include "dist_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include<vector>
namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class MultiHierarchicalNSW : public AlgorithmInterface<dist_t> { //继承AlgorithmInterface
    public:
        static const tableint max_update_element_locks = 65536;
        MultiHierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        MultiHierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements); //加载索引
        }
        //构造函数
        MultiHierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;      

            flag=0;
            num_deleted_ = 0;
            data_size_ = s->get_data_size(); //data_size_ = dim * sizeof(data type)
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;     
            maxM0_ = M_ * 2;   //maxM0设为2*M
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);
            //修改  第零层只存储原始向量与id                                                                        //边就是邻居的id
             size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint); // 每条边用4个字节，后面的字节记录共有多少个邻居
            // size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype); // 边的内存 + point本身的内存 + point id的内存
            size_data_per_element_ =  sizeof(linklistsizeint)+ data_size_ + sizeof(labeltype); 
            
            // offsetData_ = size_links_level0_;   //偏移值，用于获得指定id元素的data地址
            // label_offset_ = size_links_level0_ + data_size_; // 用于获得指定id元素的label地址
            
            //修改 
            offsetData_ = 0 ;   //偏移值，用于获得指定id元素的data地址
            label_offset_ = sizeof(linklistsizeint) + data_size_; // 用于获得指定id元素的label地址
            offsetLevel0_ = 0;   //偏移量，取得指定id元素的邻居id

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);  //给0层开辟内存
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);    //建立VisitedListPool，visited_list设为1

            //initializations for special treatment of the first node       
            enterpoint_node_ = -1;
            maxlevel_ = -1;
            temp_dist_compute=0;
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);  //二维数组 存放非0层的边
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint); //maxM_*4+4，后四个字节存储邻居数量
            mult_ = 1 / log(1.0 * M_);  
            revSize_ = 1.0 / mult_;   //层数因子
            
        }

        struct CompareByFirst { //仿函数 实现了元素根据距离比较大小
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~MultiHierarchicalNSW() { 

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }
        size_t flag;
        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t num_deleted_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_; 
        std::mutex print;

        // Locks to prevent race condition during update/insert of an element at same time. //设置锁控制多线程插入/更新/查询并行
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;
        
        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;
        tableint currObj_enter ;
        size_t data_size_;
        unsigned long long temp_dist_compute;
        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const { // 根据internal_id获取元素id
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const { //根据internal_id写入元素label
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer,unsigned long long &dist_coumputes) {

            // visitedList包括数组和数，visited_array 是数组，tag是数 
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates存储每一层距离datapoint最近的ef个邻居，对应于论文中的动态列表W， 元素按距离大小降序排列
            // candidateSet存储候选元素，对应于动态列表中的C  注意里面dist是真实距离的相反数。
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
            
            // 计算data_point(query) 到enterpoint的距离，结果保存在dist中
            dist_t lowerBound; 
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                dist_coumputes+=1;
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;  // lowerBound存储当前到datapoint的最近距离
                candidateSet.emplace(-dist, ep_id);//注意是-dist 
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id); //若该节点已删除，则标记为-lowerBound
            }
            visited_array[ep_id] = visited_array_tag; // enterpoint加入visitedList

            //candidateSet弹出顶部节点，与当前最远元素的距离比较，若胜出则继续，失败搜索结束。
            while (!candidateSet.empty()) {

                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) { //取出的
                    break;
                }
                candidateSet.pop();
                
                tableint curNodeNum = curr_el_pair.second; //获得弹出的节点的id
                
                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]); //上锁

                // 获取当前元素的邻居
                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1); // datal表示当前元素第一个邻居的label
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0); //主动缓存技术，加速操作，将数据从内存预取进cache
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
                // 对于layer层中的当前元素的每一个邻居candidate
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue; // 如果该邻居节点candidate已经访问过，无操作，循环次数加1
                    visited_array[candidate_id] = visited_array_tag; //没访问过，将其加入visitedList
                    char *currObj1 = (getDataByInternalId(candidate_id)); //取出该节点

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_); //计算其与查询点的距离
                    dist_coumputes+=1;
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {  //对应论文中:如果distance(e,q)<distance(f,q) || |W|<ef
                        candidateSet.emplace(-dist1, candidate_id); //插入候选集
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))  //如果此节点未被标记删除，插入动态结果集
                            top_candidates.emplace(dist1, candidate_id); 

                        if (top_candidates.size() > ef_construction_) //如果动态结果集满了，弹出队首元素
                            top_candidates.pop();

                        if (!top_candidates.empty())  
                            lowerBound = top_candidates.top().first; //更新lowerBound为队首的距离
                      
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);//释放VisitedList
            
            return top_candidates; // 返回动态列表，也就是返回layer层中距离q最近的ef个邻居
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer2(tableint ep_id, const void *data_point, int layer,DistList *dl,unsigned long long &dist_compute) { //加入float *dist_array

            // visitedList包括数组和数，visited_array 是数组，tag是数  float *dist_array
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            float *dist_array = dl->dists;
            auto *record = &(dl->record);

            // top_candidates存储每一层距离datapoint最近的ef个邻居，对应于论文中的动态列表W， 元素按距离大小降序排列
            // candidateSet存储候选元素，对应于动态列表中的C  注意里面dist是真实距离的相反数。
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
            
            // 计算data_point(query) 到enterpoint的距离，结果保存在dist中
            dist_t lowerBound; 
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                dist_compute+=1;
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;  // lowerBound存储当前到datapoint的最近距离
                candidateSet.emplace(-dist, ep_id);//注意是-dist 
                //dist_array[ep_id] = dist; //修改 加入dist_array
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id); //若该节点已删除，则标记为-lowerBound
            }
            visited_array[ep_id] = visited_array_tag; // enterpoint加入visitedList

            //candidateSet弹出顶部节点，与当前最远元素的距离比较，若胜出则继续，失败搜索结束。
            while (!candidateSet.empty()) {

                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) { //取出的
                    break;
                }
                candidateSet.pop();
                
                tableint curNodeNum = curr_el_pair.second; //获得弹出的节点的id
                
                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]); //上锁

                // 获取当前元素的邻居
                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1); // datal表示当前元素第一个邻居的label
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0); //主动缓存技术，加速操作，将数据从内存预取进cache
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
                // 对于layer层中的当前元素的每一个邻居candidate
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue; // 如果该邻居节点candidate已经访问过，无操作，循环次数加1
                    visited_array[candidate_id] = visited_array_tag;
                    dist_t dist1; //修改 
                    float temp_d = dist_array[candidate_id];
                    //dist_compute+=1;
                    if( temp_d < 0){
                        char *currObj1 = (getDataByInternalId(candidate_id)); //取出该节点                       
                        dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_); //计算其与查询点的距离
                        dist_compute+=1;
                        dist_array[candidate_id] = dist1; //没访问过，将其距离加入visitedList
                       // dl->record.push_back(candidate_id);
                       record->push_back(candidate_id);
                       //record->emplace(candidate_id);
                    }else{
                        dist1 = temp_d;
                    }                   

                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {  //对应论文中:如果distance(e,q)<distance(f,q) || |W|<ef
                        candidateSet.emplace(-dist1, candidate_id); //插入候选集
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))  //如果此节点未被标记删除，插入动态结果集
                            top_candidates.emplace(dist1, candidate_id); 

                        if (top_candidates.size() > ef_construction_) //如果动态结果集满了，弹出队首元素
                            top_candidates.pop();

                        if (!top_candidates.empty())  
                            lowerBound = top_candidates.top().first; //更新lowerBound为队首的距离
                      
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);//释放VisitedList
            
            return top_candidates; // 返回动态列表，也就是返回layer层中距离q最近的ef个邻居
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

// #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//                 _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//                 _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
// #endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                     _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                     _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, 
//                                  _MM_HINT_T0);////////////
// #endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
// #ifdef USE_SSE
//                             _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                          offsetLevel0_,///////////
//                                          _MM_HINT_T0);////////////////////////
// #endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                    
                            
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        void insertSort(std::vector<std::pair<dist_t, tableint>>&cp) const{
            int length = cp.size();
            auto temp = cp.back();
            auto temp_dist = temp.first;
            
            int j = length - 1;    //每轮有序区的最后元素的位置，也是有序区最大的元素位置
                while (j >= 0 && cp[j].first >= temp_dist)  //遍历有序区没有越界，且有序区元素大于待插入元素
                {
                    cp[j + 1] = cp[j];   //将有序区该位置的元素往后移一位
                    j--;  // 待遍历的位置往前移动一位，以便下次比较
                }
                cp[j + 1] = temp;  //遍历到最后，或者遍历到有序区某位置元素比待插入元素小时，将待插入元素插入到该位置    
        }    


        template <bool has_deletions, bool collect_metrics=false>
        //std::vector<std::priority_queue<std::pair<dist_t, labeltype>, std::vector<std::pair<dist_t, labeltype>>, CompareByFirst>>
        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
        searchBaseLayerST2(tableint ep_id, const void *data_point, std::vector<size_t> efsearchs,size_t k) const { //size_t ef
            VisitedList *vl = visited_list_pool_->getFreeVisitedList(); //std::vector<int> efsearchs
            vl_type *visited_array = vl->mass;  // visited_array记录邻居被访问过的节点
            vl_type visited_array_tag = vl->curV;

            VisitedList *v2 = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array2 = v2->mass;  // visited_array2记录自身被访问过的节点
            vl_type visited_array_tag2 = v2->curV;

            std::vector<std::pair<dist_t, tableint>> cp; //candidates pool数组
           // std::vector<std::priority_queue<std::pair<dist_t, labeltype>, std::vector<std::pair<dist_t, labeltype>>, CompareByFirst>> top_candidates_list;
           std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> top_candidates_list;
            
            //fsearchs.push_back(ef);
            int max_ef = efsearchs.back();
            cp.reserve(max_ef+50); //预设cp数组大小
            dist_t lowerBound; 
            // if (!has_deletions || !isMarkedDeleted(ep_id)) {
            //     dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            //     lowerBound = dist;
            //     top_candidates.emplace(dist, ep_id);
            //     candidate_set.emplace(-dist, ep_id);
            // } else {
            //     lowerBound = std::numeric_limits<dist_t>::max();
            //     candidate_set.emplace(-lowerBound, ep_id);
            // }

            // visited_array[ep_id] = visited_array_tag;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                // top_candidates.emplace(dist, ep_id);
                // candidate_set.emplace(-dist, ep_id);
                cp.push_back(std::make_pair(dist, ep_id));
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                // candidate_set.emplace(-lowerBound, ep_id);
                cp.push_back(std::make_pair(lowerBound, ep_id));
            }
            visited_array[ep_id] = visited_array_tag; //将入口点设为已访问
            visited_array2[ep_id] = visited_array_tag2;
            
            int *data = (int *) get_linklist0(ep_id);  //扫描入口点的邻居，符合条件的加入cp中
            size_t size = getListCount((linklistsizeint*)data);
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                visited_array2[candidate_id] = visited_array_tag2;

                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                if (cp.size() < efsearchs[efsearchs.size()-1] || lowerBound > dist) {
                    //candidate_set.emplace(dist, candidate_id);
                    cp.push_back(std::make_pair(dist, candidate_id));
                    //std::sort(cp.begin(),cp.end(),CompareByFirst());
                    insertSort(cp);
                if (cp.size() > max_ef) //缩减cp数组长度为max_ef
                        cp.resize(max_ef);                           
                lowerBound = cp.back().first;
                }
            }

            for(int i=0;i<efsearchs.size();i++){
                size_t count = 0;
                while (count < efsearchs[i]) {    // ef = 2  0 1 2
                    for(int it=0;it<efsearchs[i];it++){ //迭代器改成下标
                        if(!(visited_array[cp[it].second] == visited_array_tag)){//还未访问过
                            count=0;                            
                            std::pair<dist_t, tableint>  current_node_pair =cp[it];
                            tableint current_node_id = current_node_pair.second;
                            visited_array[current_node_id] = visited_array_tag;  //设置该节点为已访问

                            int *data = (int *) get_linklist0(current_node_id); //获得当前节点的邻居
                            size_t size = getListCount((linklistsizeint*)data);
        //                   bool cur_node_deleted = isMarkedDeleted(current_node_id);
                            if(collect_metrics){   //统计距离计算次数
                                metric_hops++;
                                metric_distance_computations+=size;
                            }
                            
// #ifdef USE_SSE
//                             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//                             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                             _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//                             _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
// #endif

                            for (size_t j = 1; j <= size; j++) {
                                int candidate_id = *(data + j);
            //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                             _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                             _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                                         _MM_HINT_T0);////////////
// #endif
                                if (!(visited_array2[candidate_id] == visited_array_tag2)) { //该节点被访问过

                                    visited_array2[candidate_id] = visited_array_tag2;

                                    char *currObj1 = (getDataByInternalId(candidate_id));
                                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                                    if (cp.size() < efsearchs[efsearchs.size()-1] || lowerBound > dist) {
                                       //candidate_set.emplace(dist, candidate_id);
                                       cp.push_back(std::make_pair(dist, candidate_id));
                                       //std::sort(cp.begin(),cp.end(),CompareByFirst());
                                       insertSort(cp);
// #ifdef USE_SSE
//                             _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                          offsetLevel0_,///////////
//                                          _MM_HINT_T0);////////////////////////
// #endif
                                        // if (!has_deletions || !isMarkedDeleted(candidate_id)) 不考虑删除节点
                                        //     top_candidates.emplace(dist, candidate_id);
                                        
                                        if (cp.size() > max_ef) //缩减cp数组长度为max_ef
                                            cp.resize(max_ef);
                                             
                                        lowerBound = cp.back().first;
                                    }
                                }
                            }
                            break;
                        }else{
                            count++;
                        }                           
                    }                           
                }
                
                //std::priority_queue<std::pair<dist_t, labeltype>, std::vector<std::pair<dist_t, labeltype>>, CompareByFirst> top_candidate;
                std::priority_queue<std::pair<dist_t, labeltype>> top_candidate;
                for(int j=0;j<k;j++){ //截取前k个结果
                    top_candidate.push(std::make_pair(cp[j].first, getExternalLabel(cp[j].second)));
                }
                top_candidates_list.push_back(top_candidate);
            }

            visited_list_pool_->releaseVisitedList(vl);
            visited_list_pool_->releaseVisitedList(v2);
            return top_candidates_list;
        }
        
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,unsigned long long *dist_compute = NULL) {
            if (top_candidates.size() < M) { //此时直接返回即可，不用选择近邻
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest; //
            std::vector<std::pair<dist_t, tableint>> return_list; 
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }
            // j入队列
            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first; //当前点与查询点的距离 
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {   //当前点与已经选择的点间的距离
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    *dist_compute+=1;
                    if (curdist < dist_to_query) { //
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        void getNeighborsByHeuristic3(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,DistList *dl,unsigned long long *dist_compute = NULL) {
            if (top_candidates.size() < M) { //此时直接返回即可，不用选择近邻
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest; 
            std::vector<std::pair<dist_t, tableint>> return_list; 
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);//queue_closest 存放的距离的负值
                top_candidates.pop();
            }
            // j入队列
            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first; //当前点与查询点的距离 变成正值
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {   //当前点与已经选择的点间的距离
                    dist_t curdist;
                    tableint id1,id2;
                    //auto *dist_map = &(dl->dists_pair_map);
                    if(second_pair.second < curent_pair.second ){
                        id1 = second_pair.second ;
                        id2 = curent_pair.second ;
                    }else{
                        id1 = curent_pair.second ;
                        id2 = second_pair.second ;
                    }
                    std::string s = std::to_string(id1)+std::to_string(id2);
                    // curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                    //                      getDataByInternalId(curent_pair.second),
                    //                      dist_func_param_);
                    //     //dl->dists_pair_map->insert(std::make_pair(std::make_pair(id1,id2),curdist));
                    //     *dist_compute += 1;
                    auto pos = dl->dists_pair_map->find(std::make_pair(id1,id2));
                    //auto pos = dl->dists_pair_map->find(s);
                   //auto pos = dl->dists_pair_map->find(id1);
                    //std::cout<<dl->dists_pair_map->size()<<std::endl;
                    if(pos != dl->dists_pair_map->end()){
                        // curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                        //                  getDataByInternalId(curent_pair.second),
                        //                  dist_func_param_);
                        // *dist_compute += 1;
                       curdist = (*pos).second ;
                       //std::cout<<(*pos).second<<std::endl;
                    }else{
                         curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);
                         
                       dl->dists_pair_map->insert(std::pair(std::pair(id1,id2),curdist));
                       // dl->dists_pair_map->insert(std::pair(s,curdist));
                        
                        *dist_compute += 1;
                    }
                    if (curdist < dist_to_query) { //
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);//top_candidates 存放的距离的正值
            }
        }

        //修改3 获取邻居列表函数
        // linklistsizeint *get_linklist0(tableint internal_id) const { //在第零层，根据internal_id 返回邻居数组
        //     return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        // };

        // linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        //     return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        // };

        linklistsizeint *get_linklist0(tableint internal_id) const { //修改3 在第零层，根据internal_id 返回邻居数组
            return (linklistsizeint *) (linkLists_[internal_id]);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {//修改3 在第零层，根据internal_id 返回邻居数组
            return (linklistsizeint *) (linkLists_[internal_id]);
        };
        
        linklistsizeint *get_linklist(tableint internal_id, int level) const {  //指定id,获得其第level层的邻居
            return (linklistsizeint *) (linkLists_[internal_id] + size_links_level0_ + (level-1) * size_links_per_element_);//后面是偏移量，通过其获得第level层的邻居起始地址
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        //将给定节点与其邻居节点建立连接
        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate,unsigned long long *dist_compute = NULL ) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_,dist_compute); //从top_candidates中启发式选择M_个近邻,选择结果放在selectedNeighbors
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors; 
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second); 
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            //将选择的邻居写入近邻列表
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                //修改3 设置邻居数目
                // setListCount(ll_cur,selectedNeighbors.size());
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) { 
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }
            //检查其邻居是否边数超载，对超载的邻居也进行启发式选边
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) { //该邻居的近邻数还未达到M，将该节点加入其近邻表
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        *dist_compute+=1;
                        // Heuristic:   
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                            *dist_compute+=1;
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax,dist_compute);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }


        //将给定节点与其邻居节点建立连接
        tableint mutuallyConnectNewElement2(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate,DistList *dl,unsigned long long *dist_compute = NULL) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic3(top_candidates, M_,dl,dist_compute); //从top_candidates中启发式选择M_个近邻,选择结果放在selectedNeighbors
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors; 
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second); 
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            //将选择的邻居写入近邻列表
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                //修改3 设置邻居数目
                // setListCount(ll_cur,selectedNeighbors.size());
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1); //data 为cur节点的邻居列表
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) { 
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }
            //检查其邻居是否边数超载，对超载的邻居也进行启发式选边
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) { //该邻居的近邻数还未达到M，将该节点加入其近邻表
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        *dist_compute+=1;
                        // Heuristic:   
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) { //
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                            *dist_compute+=1;
                        }

                        getNeighborsByHeuristic3(candidates, Mcurmax,dl,dist_compute);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }
        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    num_deleted_ += 1;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        //get linklist count 获得邻居数量
        unsigned short int getListCount(linklistsizeint * ptr) const { 
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        unsigned long long addPoint(const void *data_point, labeltype label) {
            unsigned long long dist_coumputes=0;
            addPoint(data_point, label,-1,dist_coumputes);
            return dist_coumputes;
        }

        void addPoint2(const void *data_point, labeltype label,float *dist_array) { //改进后的添加节点
            // print.lock();
            // std::cout<<"dist_array:  "<<&dist_array<<" \ti:  "<<label<<std::endl;
            // print.unlock();
            //addPoint2(data_point, label,-1,dist_array);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");
             unsigned long long dd;
            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level,dd);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };



        tableint addPoint(const void *data_point, labeltype label, int level,unsigned long long &dist_coumputes) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);//cur_element_count_guard_ 增员锁，当新增向量时，用于更新当前向量总数和<label，id>映射，临界区很小
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c; //label_lookup_  记录了<label，id>映射
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);//邻居更新锁
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);//邻居列表锁
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel; //设置当前元素的层数


            std::unique_lock <std::mutex> templock(global); //global全局锁，这个锁仅在新增向量流程中，且所在层大于当前最大层时
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock(); //不更新最大层数
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            //待修改 第零层要去掉邻居列表，只留原始向量和label，考虑内存对齐
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            // if (curlevel) {
            //     linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            //     if (linkLists_[cur_c] == nullptr)
            //         throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            //     memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            // }
            // Initialisation of the linkLists_ 初始化邻居表 修改 将第0层邻居表也加进来
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1 + size_links_level0_);//这个1字节是干什么的？
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1 + size_links_level0_);            

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) { //从最高层向倒数第二层搜索

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    dist_coumputes+=1;
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]); //lock currObj 邻居列表锁
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i]; 
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                dist_coumputes+=1;
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level,dist_coumputes);  //通过searchBaseLayer()获得top_candidates
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false,&dist_coumputes); ////根据top_candidates连接元素
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        tableint get_cur_id(labeltype label){
            tableint cur_c = 0;
            {
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                // auto search = label_lookup_.find(label);
                // if (search != label_lookup_.end()) {
                //     tableint existingInternalId = search->second;
                //     templock_curr.unlock();

                //     std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                //     if (isMarkedDeleted(existingInternalId)) {
                //         unmarkDeletedInternal(existingInternalId);
                //     }
                //     updatePoint(data_point, existingInternalId, 1.0);
                    
                //     return existingInternalId;
                // }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                //label_lookup_[label] = cur_c;
                return cur_c;
            }
        }
        unsigned long long addPoint2(const void *data_point, labeltype label, int level,DistList *dl,tableint cur,int flags) { //修改加入float *dist_array
           // std::cout<< omp_get_num_threads  float *dist_array
            unsigned long long dist_compute = 0;
            tableint cur_c = cur;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                 std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                // auto search = label_lookup_.find(label);
                // if (search != label_lookup_.end()) {
                //     tableint existingInternalId = search->second;
                //     templock_curr.unlock();

                //     std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                //     if (isMarkedDeleted(existingInternalId)) {
                //         unmarkDeletedInternal(existingInternalId);
                //     }
                //     updatePoint(data_point, existingInternalId, 1.0);
                    
                //     return existingInternalId;
                // }

                // if (cur_element_count >= max_elements_) {
                //     throw std::runtime_error("The number of elements exceeds the specified limit");
                // };

                // cur_c = cur_element_count;
                if(flags){
                    cur_element_count = cur_c + 1;
                }                 
                label_lookup_[label] = cur_c;
            }
            float *dist_array = dl->dists;
            auto *record = &(dl->record);
            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]); //cur_c节点增量构建坑位锁 65536个
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]); //cur_c节点邻居表锁
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel; //设置当前元素的层数


            std::unique_lock <std::mutex> templock(global); //全局锁 仅在新增向量流程中，且所在层大于当前最大层时锁住
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock(); //全局锁解锁
            tableint currObj = enterpoint_node_;   //enterpoint_node_ 初始为-1
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_); //该元素节点先用0初始化填充

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            // if (curlevel) {  //若该节点在非零层，进行初始化
            //     linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            //     if (linkLists_[cur_c] == nullptr)
            //         throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            //     memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            // }
            //修改3 初始化邻居列表
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1 + size_links_level0_);//这个1字节是干什么的？
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1 + size_links_level0_);    

            if ((signed)currObj != -1) {  //非索引中第一个元素

                if (curlevel < maxlevelcopy) {  //从最顶层搜索到当前要插入的上一层

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    dist_compute +=1;
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]); //currObj节点邻居更新锁 
                            data = get_linklist(currObj,level); //data数组存放了邻居id
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];  // cand代表第i个邻居的id
                                if (cand < 0 || cand > max_elements_) //判断id是否越界
                                    throw std::runtime_error("cand error");
                            
                                dist_t d;
                                float temp_d = dist_array[cand];
                                
                               // auto ssize = dl->record.size();
                                if(temp_d < 0){
                                     d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                     dist_compute +=1;
                                     dist_array[cand] = d;
                                    // dl->record.push_back(cand);
                                     record->push_back(cand);
                                     //record->emplace(cand);
                                     //std::cout<<"record size   :"<<dl->record.size()<<std::endl;
                                }else{
                                    d = temp_d;
                                }
                                //dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) { //从当前层到最底层
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer2(
                            currObj, data_point, level,dl,dist_compute);  //通过searchBaseLayer()获得top_candidates
                    // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                    //         currObj, data_point, level);  //通过searchBaseLayer()获得top_candidates
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    //currObj = mutuallyConnectNewElement2(data_point, cur_c, top_candidates, level, false,dl,&dist_compute); ////根据top_candidates连接元素
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false,&dist_compute); ////根据top_candidates连接元素
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return dist_compute;
        };

        // void getenterpoint_node(tableint  enterpoint_node_){
        //         tableint currObj = enterpoint_node_;
        //         return currObj;
        // }
        
        //std::priority_queue<std::pair<dist_t, labeltype >>
        //std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>>
        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
        searchKnn2(const void *query_data, size_t k, std::vector<size_t> efsearchs,tableint enterpoint_) const { 
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            //if (cur_element_count == 0) return result;  当前元素为0，直接返回结果

           // tableint currObj = enterpoint_node_;
            tableint currObj = enterpoint_;
            //tableint currObj = 50; //入口点固定
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) { //从顶层逐层往下搜索，直到第0层，得到0层的入口点
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand; 
                            changed = true;
                        }
                    }
                }
            }
             std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> top_candidates_list;
             //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_;
            
             //std::cout<<"currObj"<<currObj<<endl;
            if (num_deleted_) {
                // top_candidates_=searchBaseLayerST<true,true>(
                //         currObj, query_data, std::max(ef_, k));  
                top_candidates_list=searchBaseLayerST2<true,true>(currObj, query_data, efsearchs, k);  
            }
            else{
                // top_candidates_=searchBaseLayerST<false,true>(
                //         currObj, query_data, std::max(ef_, k)); //currObj 入口点
                top_candidates_list=searchBaseLayerST2<false,true>(currObj, query_data, efsearchs,k); 
            }

            // for(int i=0;i<efsearchs.size();i++){
            //     while (top_candidates_list[i].size() > k) {
            //         top_candidates_list[i].pop();
            //      }
            // }
            // while (top_candidates_.size() > k) {
            //     top_candidates_.pop();
            // }
            // while (top_candidates_.size() > 0) {  //将top_candidates放入result中返回
            //     std::pair<dist_t, tableint> rez = top_candidates_.top();
            //     result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            //     top_candidates_.pop();
            // }
            // top_candidates_list.push_back(result);
            return top_candidates_list;
        };
        
        tableint get_curr(){
                return enterpoint_node_;
        }

        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k,tableint enterpoint_) const { 
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            //tableint currObj = enterpoint_node_;
            tableint currObj = enterpoint_;
            //dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_), dist_func_param_);
            for (int level = maxlevel_; level > 0; level--) { //从顶层逐层往下搜索，直到第0层，得到0层的入口点
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand; 
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (num_deleted_) {
                top_candidates=searchBaseLayerST<true,true>(
                        currObj, query_data, std::max(ef_, k));  
                
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(
                        currObj, query_data, std::max(ef_, k)); //currObj 入口点               
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {  //将top_candidates放入result中返回
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}
