set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC")
find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_VERSION: ${Python_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS} 
    )

add_library(websocket STATIC
  websocket_client.cc
  websocket_server.cc
)
target_link_libraries(websocket PUBLIC decoder ${Python_LIBRARIES})

