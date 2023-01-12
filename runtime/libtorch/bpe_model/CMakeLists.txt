cmake_minimum_required(VERSION 3.12)
project(test)

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_VERSION: ${Python_INCLUDE_DIRS}")
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS} 
    )
# 生成目标文件
add_executable(test test.cpp)
# 链接库
target_link_libraries(test ${Python_LIBRARIES})

