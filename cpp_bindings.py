import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer

os.chdir(os.path.dirname(os.path.realpath(__file__)))

lib = ctypes.CDLL('./build/libpybindings.so')

lib.run.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # d
    ctypes.c_double,  # r
    ctypes.c_double,  # W
    ctypes.c_double,  # L
    ctypes.c_double,  # a_w
    ctypes.c_double,  # a_l
    ctypes.c_int,  # max_iters
    ctypes.c_double,  # exit_tol
    ctypes.c_int,  # ipopt_max_iter
    ctypes.c_double,  # ipopt_tol
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')  # result
]
lib.run.restype = None

def solve(
    n_persons: int,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    r: float,
    W: float = 1.0,
    L: float = 0.0,
    a_w: float = 0.0,
    a_l: float = 0.0,
    max_iters: int = 200,
    exit_tol: float = 0.001,
    ipopt_max_iters: int = 200,
    ipopt_tol: float = 0.001
) -> np.ndarray:
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d))
    result = np.empty((2, n_persons))
    lib.run(
        n_persons, A, alpha, B, beta, theta, d, r,
        W, L, a_w, a_l,
        max_iters, exit_tol, ipopt_max_iters, ipopt_tol,
        result
    )
    return result.T


lib.prod_F.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Ks
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Kp
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # s_out
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # p_out
]
lib.prod_F.restype = None

def prod_F(
    n_persons: int,
    Ks: np.ndarray,
    Kp: np.ndarray,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(Ks) == len(Kp))
    s_out, p_out = np.empty(n_persons), np.empty(n_persons)
    lib.prod_F(n_persons, A, alpha, B, beta, theta, Ks, Kp, s_out, p_out)
    return s_out, p_out


lib.get_payoffs.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # d
    ctypes.c_double,  # r
    ctypes.c_double,  # W
    ctypes.c_double,  # L
    ctypes.c_double,  # a_w
    ctypes.c_double,  # a_l
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Ks
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Kp
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')  # payoffs_out
]
lib.get_payoffs.restype = None

def get_payoffs(
    n_persons: int,
    Ks: np.ndarray,
    Kp: np.ndarray,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    r: float,
    W: float = 1.0,
    L: float = 0.0,
    a_w: float = 0.0,
    a_l: float = 0.0
):
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d) == len(Ks) == len(Kp))
    payoffs = np.empty(n_persons)
    lib.get_payoffs(
        n_persons, A, alpha, B, beta, theta,
        d, r,
        W, L, a_w, a_l,
        Ks, Kp,
        payoffs
    )
    return payoffs


lib.run_variable_r.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # d
    ctypes.c_double,  # r0
    ctypes.c_double,  # c
    ctypes.c_int,  # max_iters
    ctypes.c_double,  # exit_tol
    ctypes.c_int,  # ipopt_max_iter
    ctypes.c_double,  # ipopt_tol
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')  # result
]
lib.run_variable_r.restype = None

def solve_variable_r(
    n_persons: int,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    r0: float,
    c: float,
    max_iters: int = 200,
    exit_tol: float = 0.001,
    ipopt_max_iters: int = 200,
    ipopt_tol: float = 0.001
) -> np.ndarray:
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d))
    result = np.empty((2, n_persons))
    lib.run_variable_r(
        n_persons, A, alpha, B, beta, theta, d, r0, c,
        max_iters, exit_tol, ipopt_max_iters, ipopt_tol,
        result
    )
    return result.T


lib.get_payoffs_variable_r.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # d
    ctypes.c_double,  # r0
    ctypes.c_double,  # c
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Ks
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Kp
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')  # payoffs_out
]
lib.get_payoffs_variable_r.restype = None

def get_payoffs_variable_r(
    n_persons: int,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    r0: float,
    c: float,
    Ks: np.ndarray,
    Kp: np.ndarray
):
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d) == len(Ks) == len(Kp))
    payoffs = np.empty(n_persons)
    lib.get_payoffs_variable_r(n_persons, A, alpha, B, beta, theta, d, r0, c, Ks, Kp, payoffs)
    return payoffs



if __name__ == '__main__':
    # just a simple test
    
    n_persons = 3
    ones = np.ones(n_persons, dtype=np.float64)
    A = ones * 1.0
    alpha = ones * 0.5
    B = ones * 1.0
    beta = ones * 0.5
    theta = ones * 0.0

    d = ones * 1.0
    r = 0.01

    K = solve(
        n_persons,
        A,
        alpha,
        B,
        beta,
        theta,
        d,
        r
    )
    print(K)

    # need to copy since cpp expects contiguous memory
    Ks, Kp = K[:, 0].copy(), K[:, 1].copy()
    print(
        prod_F(
            n_persons,
            Ks,
            Kp,
            A,
            alpha,
            B,
            beta,
            theta
        )
    )

    print(
        get_payoffs(
            n_persons,
            Ks,
            Kp,
            A,
            alpha,
            B,
            beta,
            theta,
            d,
            r,
        )
    )


    r0 = 0.01
    c = 0.1

    K = solve_variable_r(
        n_persons,
        A,
        alpha,
        B,
        beta,
        theta,
        d,
        r0,
        c
    )
    print(K)

    Ks, Kp = K[:, 0].copy(), K[:, 1].copy()
    print(
        prod_F(
            n_persons,
            Ks,
            Kp,
            A,
            alpha,
            B,
            beta,
            theta
        )
    )

    print(
        get_payoffs_variable_r(
            n_persons,
            Ks,
            Kp,
            A,
            alpha,
            B,
            beta,
            theta,
            d,
            r0,
            c
        )
    )
