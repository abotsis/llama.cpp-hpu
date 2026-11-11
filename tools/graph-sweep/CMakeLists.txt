# llama-graph-sweep: fill the graph recipe cache of a backend that compiles one graph per shape (SYNAPSE)

set(TARGET llama-graph-sweep)

add_executable(${TARGET} graph-sweep.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
