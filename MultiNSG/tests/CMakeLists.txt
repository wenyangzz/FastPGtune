set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_multi_graph_builder test_multi_graph_builder.cpp)
target_link_libraries(test_multi_graph_builder ${PROJECT_NAME})

add_executable(nsg_graph_similarity nsg_graph_similarity.cpp)
target_link_libraries(nsg_graph_similarity ${PROJECT_NAME})

add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent ${PROJECT_NAME})

add_executable(test_1 test_1.cpp)
target_link_libraries(test_1 ${PROJECT_NAME})

add_executable(test_search test_search.cpp)
target_link_libraries(test_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

