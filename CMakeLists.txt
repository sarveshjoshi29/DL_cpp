cmake_minimum_required(VERSION 3.10)
# The executable will be placed in C:/MyPrograms/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:/CodingWorkspace/C++/DL_cpp/exe")
# Set the project name
project(DL_cpp)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nn_lib
    src/Batch_Handler.cpp
    src/gd_momentum.cpp
    src/Dense.cpp
    src/rmsprop.cpp
    src/wrapper.cpp
    src/Adam.cpp
    src/Dropout.cpp
)

target_include_directories(nn_lib 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test
    main/main.cpp 
)

target_link_libraries(test nn_lib)

	