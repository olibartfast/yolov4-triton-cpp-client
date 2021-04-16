## C++ Triton YoloV4 client 
Developed to infer the model deployed in Nvidia Triton Server like in [Isarsoft yolov4-triton-tensorrt repo](https://github.com/isarsoft/yolov4-triton-tensorrt), inference part based on [Wang-Xinyu tensorrtx Yolov4 code](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4) and communication with server based on [Triton image client](https://github.com/triton-inference-server/server/blob/master/docs/client_examples.md#image-classification-example) example

## Build or download client libraries
https://github.com/triton-inference-server/server/blob/master/docs/client_libraries.md


## Dependencies
protobuf, grpc++(you can use libraries builded inside server folder)
cuda
rapidjson
opencv

## Build and compile
* mkdir build 
* cd build 
* cmake .. 
* make

## How to run
* ./yolov4-triton-cpp-client  --video=/path/to/video/videoname.format
* ./yolov4-triton-cpp-client  --help for all available parameters

### Video test
https://youtu.be/VsENXGMNlhA

IN PROGRESS AND TO IMPROVE...
