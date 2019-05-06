from .hats cimport hat, hatp
import numpy as np

cpdef hhp(const double[:] xx, const size_t n):

    cdef:
        size_t xi, i, xlen = len(xx)
    h_ = np.empty((xlen, n), dtype=np.float64)
    hp_ = np.empty((xlen, n), dtype=np.float64)

    cdef:
        double[:, :] h = h_
        double[:, :] hp = hp_

    for xi in range(xlen):
        for i in range(1, n+1):
            h[xi, i-1] = hat(xx[xi], i, n)
            hp[xi, i-1] =  hatp(xx[xi], i, n)

    return h_, hp_

