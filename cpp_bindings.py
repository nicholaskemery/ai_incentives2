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
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # r
    ctypes.c_int,  # max_iters
    ctypes.c_double,  # exit_tol
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
    r: np.ndarray,
    max_iters: int,
    exit_tol: float
) -> np.ndarray:
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d))
    result = np.empty((2, n_persons))
    lib.run(n_persons, A, alpha, B, beta, theta, d, r, max_iters, exit_tol, result)
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
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    Ks: np.ndarray,
    Kp: np.ndarray
) -> np.ndarray:
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta))
    s_out, p_out = np.empty(n_persons), np.empty(n_persons)
    lib.prod_F(n_persons, A, alpha, B, beta, theta, Ks, Kp, s_out, p_out)
    return np.stack((s_out, p_out), 1)


lib.get_payoffs.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # A
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # alpha
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # B
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # beta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # theta
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # d
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # r
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Ks
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # Kp
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')  # payoffs_out
]
lib.get_payoffs.restype = None

def get_payoffs(
    n_persons: int,
    A: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    Ks: np.ndarray,
    Kp: np.ndarray
):
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d))
    payoffs = np.empty(n_persons)
    lib.get_payoffs(n_persons, A, alpha, B, beta, theta, d, r, Ks, Kp, payoffs)
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
    r = ones * 0.01
    max_iters = 100
    exit_tol = 0.001

    K = solve(
        n_persons,
        A,
        alpha,
        B,
        beta,
        theta,
        d,
        r,
        max_iters,
        exit_tol
    )
    print(K)

    # need to copy since cpp expects contiguous memory
    Ks, Kp = K[:, 0].copy(), K[:, 1].copy()
    print(
        prod_F(
            n_persons,
            A,
            alpha,
            B,
            beta,
            theta,
            Ks,
            Kp
        )
    )

    print(
        get_payoffs(
            n_persons,
            A,
            alpha,
            B,
            beta,
            theta,
            d,
            r,
            Ks,
            Kp
        )
    )
