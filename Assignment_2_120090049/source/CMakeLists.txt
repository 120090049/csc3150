cmake_minimum_required(VERSION 3.5.1)
project(CSC3150_Assignment_2)

include_directories('.')

link_libraries(pthread)

add_executable(CSC3150_Assignment_2
        ./hw2.cpp
        ./game.h)