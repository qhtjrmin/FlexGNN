#include <chrono>
#include <memory>
#include <string>
#include <cuda_runtime.h>

#include "model/supported_model.hpp"
#include "utils/print.hpp"
#include "data_structures/model_options.hpp"

ModelOptions parse_model_options(int argc, char* argv[]) {
    ArgumentParser parser;
    parser.parse(argc, argv);

    std::string input_path = parser.get_string("-input_path");
    std::string config_path = parser.get_string("-config_path");
    std::string model_name = parser.get_string("-model");
    
    std::shared_ptr<XXLConfig> config = load_config(config_path);
    
    // optional arguments
    int num_gpus = parser.get_int("-g");
    if(num_gpus <= 0) cudaGetDeviceCount(&num_gpus);

    int num_epochs = parser.get_int("-e");
    if(num_epochs <= 0) num_epochs = config->training->num_epochs;

    int fixed_h = parser.get_int("-fixed_h");
    bool is_directed = parser.get_bool("-is_directed");

    ModelOptions model_options;
    model_options.name        = model_name;
    model_options.input_path  = input_path;
    model_options.config      = config;
    model_options.num_gpus    = num_gpus;
    model_options.fixed_h     = fixed_h;
    model_options.is_directed = is_directed;
    model_options.num_epochs  = num_epochs;

    return model_options;
}

int main(int const argc, char* argv[]){

    print::program_title("FlexGNN", "Large Scale Full-Graph GNN Training");

    auto st_total = std::chrono::high_resolution_clock::now();

    // parse arguments
    ModelOptions model_options = parse_model_options(argc, argv);
    std::shared_ptr<XXLConfig> config = model_options.config;
    int num_gpus = model_options.num_gpus;

    // enable CUDA peer access
    for(int i = 0; i < num_gpus; i++){
        for(int j = 0; j < num_gpus; j++){
            if(i != j){
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    print::section_header("System Info");

    Model* model = nullptr;
    if(model_options.name == "GCN"){
        print::key_value("Model", "GCN");
        model = new LargeGCN(model_options);
    }
    else if(model_options.name == "GraphSage"){
        print::key_value("Model", "GraphSage");
        model = new LargeGraphSage(model_options);
    }
    else if(model_options.name == "GAT"){
        print::key_value("Model", "GAT");
        model = new LargeGAT(model_options);
    }
    else{
        print::error("Unknown model type: " + model_options.name);
        return 1;
    }

    // (1) input & model preparation
    print::section_header("Input Preparation");
    model->load_inputs();
    model->prepare_model();
    model->prepare_learning_parameters(config->model->layers.size(), config->model->random_seed);
    
    // (2) training preparation
    print::section_header("Training Preparation");
    auto st = std::chrono::high_resolution_clock::now();
    model->prepare_training();
    auto et = std::chrono::high_resolution_clock::now();
    print::elapsed_time("Training Preparation", st, et);
    
    print::cuda_mem_info();
    print::system_mem_info();

    // (3) run training
    model->set_workers(model_options.num_epochs);
    model->train();

    // (4) run testing
    model->test();

    // (5) print final summary
    auto et_total = std::chrono::high_resolution_clock::now();
    auto dt_total = std::chrono::duration_cast<std::chrono::seconds>(et_total - st_total).count();
    print::final_summary(dt_total);
    
    delete model;

    return 0;
}