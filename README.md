## C++ Triton YoloV4 client 
Developed to infer the model deployed in Nvidia Triton Server like in [Isarsoft yolov4-triton-tensorrt repo](https://github.com/isarsoft/yolov4-triton-tensorrt), inference part based on [Wang-Xinyu tensorrtx Yolov4 code](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4) and communication with server based on [Triton image client](https://github.com/triton-inference-server/server/blob/master/docs/client_examples.md#image-classification-example) example

## Build client libraries
https://github.com/triton-inference-server/server/blob/master/docs/client_libraries.md


## Dependencies
* Nvidia Triton Inference Server container pulled from NGC(Tested Release 21.02)
* Triton client libraries
* Protobuf, Grpc++, Rapidjson(versions according to the ones used within Triton server project. I used libraries built inside Triton Server folder)
* Cuda(Tested 11.2)
* Opencv4(Tested 4.2.0)

## Build and compile
* mkdir build 
* cd build 
* cmake -DCMAKE_BUILD_TYPE=Release .. 
* make

## How to run
* ./yolov4-triton-cpp-client  --video=/path/to/video/videoname.format
* ./yolov4-triton-cpp-client  --help for all available parameters

### Realtime inference test on video
* Inference test ran from VS Code: https://youtu.be/IUdbplJlspg
* other video inference test: https://youtu.be/VsENXGMNlhA
