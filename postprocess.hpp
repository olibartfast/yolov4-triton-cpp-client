#pragma once
#include "common.hpp"

auto
PostprocessYoloV4(
    nic::InferResult* result,
    const size_t batch_size,
    const std::vector<std::string>& output_names, const bool batching)
{
    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference  failed with error: " << result->RequestStatus()
            << std::endl;
        exit(1);
    }

    std::vector<float> detections;
    std::vector<int64_t> shape;


    float* outputData;
    size_t outputByteSize;
    for (auto outputName : output_names)
    {
        if (outputName == "prob")
        { 
            result->RawData(
                outputName, (const uint8_t**)&outputData, &outputByteSize);

            nic::Error err = result->Shape(outputName, &shape);
            detections = std::vector<float>(outputByteSize / sizeof(float));
            std::memcpy(detections.data(), outputData, outputByteSize);
            if (!err.IsOk())
            {
                std::cerr << "unable to get data for " << outputName << std::endl;
                exit(1);
            }
        }

    }

    return make_tuple(detections, shape);
}