#pragma once
#include "common.hpp"

union TritonClient
{
    TritonClient()
    {
        new (&httpClient) std::unique_ptr<nic::InferenceServerHttpClient>{};
    }
    ~TritonClient() {}

    std::unique_ptr<nic::InferenceServerHttpClient> httpClient;
    std::unique_ptr<nic::InferenceServerGrpcClient> grpcClient;
};
