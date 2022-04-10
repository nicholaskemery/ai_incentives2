#pragma once

#include <vector>
#include "solve.hpp"
#include "problem.hpp"
#include "prodFunc.hpp"

// template <typename T>
// std::vector<T> read_numpy_1d(const T* array_ptr, int array_len) {
//     std::vector<T> vec;
//     vec.reserve(array_len);
//     for (int i = 0; i < array_len; i++) {
//         vec.push_back(array_ptr[i]);
//     }
//     return vec;
// }

extern "C" {
    void run(
        int n_persons,
        double* A,
        double* alpha,
        double* B,
        double* beta,
        double* theta,
        double* d,
        double r,
        int max_iters,
        double exit_tol,
        double* result
    );
}