// ggml-synapse-worker: serves one Gaudi card to a parent process (see ggml-synapse-proxy.cpp)
extern "C" int ggml_synapse_worker_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    return ggml_synapse_worker_main(argc, argv);
}
