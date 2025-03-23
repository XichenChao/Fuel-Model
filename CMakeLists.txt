cmake_minimum_required(VERSION 3.28)
project(c___fuel)

set(CMAKE_CXX_STANDARD 17)

include_directories(/opt/homebrew/include/eigen3)

add_executable(c___fuel main.cpp)
