cmake_minimum_required(VERSION 3.00 FATAL_ERROR)

project(ai_incentives)
add_executable(main main.cpp)
add_library(lib prodFunc.hpp prodFunc.cpp problem.hpp problem.cpp solve.hpp solve.cpp)

find_package(Eigen3 REQUIRED)
find_package(ifopt REQUIRED)
target_link_libraries(lib Eigen3::Eigen ifopt::ifopt_ipopt)
target_link_libraries(main lib)
