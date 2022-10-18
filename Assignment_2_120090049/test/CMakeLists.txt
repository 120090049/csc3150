cmake_minimum_required(VERSION 3.5.1)
project(test)

include_directories('src')

link_libraries(pthread)

add_executable(out
        src/test.cpp)