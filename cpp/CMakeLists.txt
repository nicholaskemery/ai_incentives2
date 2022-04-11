cmake_minimum_required(VERSION 3.00 FATAL_ERROR)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

project(ai_incentives)

add_library(lib prodFunc.hpp prodFunc.cpp problem.hpp problem.cpp solve.hpp solve.cpp)

find_package(Eigen3 REQUIRED)
find_package(ifopt REQUIRED)
target_link_libraries(lib Eigen3::Eigen ifopt::ifopt_ipopt)
set_property(TARGET lib PROPERTY CXX_STANDARD 17)

# add_executable(main main.cpp)
# target_link_libraries(main lib)
# set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_library(pybindings SHARED pybindings.hpp pybindings.cpp)
target_link_libraries(pybindings lib)
set_property(TARGET pybindings PROPERTY CXX_STANDARD 17)
