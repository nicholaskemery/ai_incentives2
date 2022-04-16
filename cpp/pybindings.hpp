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
        double* r,
        double W,
        double L,
        double a_w,
        double a_l,
        int max_iters,
        double exit_tol,
        int ipopt_max_iter,
        double ipopt_tol,
        double* result
    );

    void prod_F(
        int n_persons,
        double* A,
        double* alpha,
        double* B,
        double* beta,
        double* theta,
        double* Ks,
        double* Kp,
        double* s_out,
        double* p_out
    );

    void get_payoffs(
        int n_persons,
        double* A,
        double* alpha,
        double* B,
        double* beta,
        double* theta,
        double* d,
        double* r,
        double W,
        double L,
        double a_w,
        double a_l,
        double* Ks,
        double* Kp,
        double* payoffs_out
    );

    void run_variable_r(
        int n_persons,
        double* A,
        double* alpha,
        double* B,
        double* beta,
        double* theta,
        double* d,
        double r0,
        double c,
        int max_iters,
        double exit_tol,
        int ipopt_max_iter,
        double ipopt_tol,
        double* result
    );

    void get_payoffs_variable_r(
        int n_persons,
        double* A,
        double* alpha,
        double* B,
        double* beta,
        double* theta,
        double* d,
        double r0,
        double c,
        double* Ks,
        double* Kp,
        double* payoffs_out
    );
}