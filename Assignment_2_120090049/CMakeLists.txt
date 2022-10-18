cmake_minimum_required(VERSION 3.5.1)
project(CSC3150_Assignment_2)

include_directories('source')

link_libraries(pthread)

add_executable(CSC3150_Assignment_2
        source/hw2.cpp
        source/game.h)