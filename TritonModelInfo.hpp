#pragma once
#include "common.hpp"


struct TritonModelInfo {
    std::string output_name_;
    std::vector<std::string> output_names_;
    std::string input_name_;
    std::string input_datatype_;
    // The shape of the input
    int input_c_;
    int input_h_;
    int input_w_;
    // The format of the input
    std::string input_format_;
    int type1_;
    int type3_;
    int max_batch_size_;

    std::vector<int64_t> shape_;

};

