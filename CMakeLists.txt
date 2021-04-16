cmake_minimum_required(VERSION 3.0)

project(yolov4-triton-cpp-client)
message(STATUS ${PROJECT_NAME}) 
message(STATUS $ENV{Triton_Build_DIR})
message(STATUS $ENV{HOME})
set(CMAKE_CXX_STANDARD 17)

set(CURL_DIR "$ENV{Triton_Build_DIR}/curl/install/lib/cmake/CURL")
set(Protobuf_DIR "$ENV{Triton_Build_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "$ENV{Triton_Build_DIR}/grpc/lib/cmake/grpc")
set(c-ares_DIR "$ENV{Triton_Build_DIR}/c-ares/lib/cmake/c-ares")
set(TRITON_DIR "$ENV{Triton_Build_DIR}/client/install/lib/cmake/TRITON")
set(TritonCommon_DIR "$ENV{Triton_Build_DIR}/client/install/lib/cmake/TritonCommon")

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TRITON REQUIRED)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/yolov4-client.cpp)
target_link_libraries(${PROJECT_NAME} 
PRIVATE TRITON::grpcclient_static
PRIVATE TRITON::httpclient_static
PRIVATE TritonCommon::triton-common-json
PRIVATE ${OpenCV_LIBS})