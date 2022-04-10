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
    r: float,
    max_iters: int,
    exit_tol: float
):
    assert(n_persons == len(A) == len(alpha) == len(B) == len(beta) == len(theta) == len(d))
    result = np.empty((n_persons, 2))
    lib.run(n_persons, A, alpha, B, beta, theta, d, r, max_iters, exit_tol, result)
    return result.T  # transpose since Eigen and numpy represent rows and columns in different order


if __name__ == '__main__':
    # just a simple test
    
    n_persons = 2
    ones = np.ones(n_persons, dtype=np.float64)
    A = ones * 1.0
    alpha = ones * 0.5
    B = ones * 1.0
    beta = ones * 0.5
    theta = ones * 0.0

    d = ones * 1.0
    r = 0.01
    max_iters = 100
    exit_tol = 0.001

    print(
        solve(
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
    )
