cmake_minimum_required(VERSION 3.00 FATAL_ERROR)

project(ai_incentives)
add_executable(main main.cpp)
add_library(lib prodFunc.hpp prodFunc.cpp problem.hpp problem.cpp)

find_package(Eigen3 REQUIRED)
target_link_libraries(lib Eigen3::Eigen)
target_link_libraries(main lib)
