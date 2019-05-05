import numpy as np


cpdef double hat(double x, size_t i_, size_t n_) except *:
    cdef:
        double i = <double>i_
        double np1 = <double>(n_+1)

    if x < ((i-1)/np1):
        return 0
    if x > ((i+1)/np1):
        return 0

    if x < (i/np1):
        return np1*x - i + 1
    if x > (i/np1):
        return i - np1*x + 1


cpdef double hatp(double x, size_t i_, size_t n_) except *:
    cdef:
        double i = <double>i_
        double np1 = <double>(n_+1)

    if x < ((i-1)/np1):
        return 0
    if x > ((i+1)/np1):
        return 0

    if x < (i/np1):
        return np1
    if x > (i/np1):
        return -np1*x


cpdef double hat_dot_pow3(double x, double[:] w, size_t n) except *:

    cdef:
        double[:] h = np.empty(n, dtype=np.float64)
        size_t i

    for i in range(1, n+1):
        h[i-1] = hat(x, i, n)

    return _hat_dot_pow3(x, w, h, n)


cpdef double hatp_dot_pow3(double x, double[:] w, size_t n) except *:

    cdef:
        double[:] h = np.empty(n, dtype=np.float64)
        size_t i

    for i in range(1, n+1):
        h[i-1] = hatp(x, i, n)

    return _hat_dot_pow3(x, w, h, n)


# TODO implement a recursive model of this that can take arbitrary powers.
cdef double _hat_dot_pow3(double x, double[:] w, double[:] h, size_t n) except *:
    cdef:
        double s = 0
        size_t i, j, k

    for i in range(1, n+1):
        for j in range(i-1, i+2):
            if 1 > j or j >= n+1:
                continue
            if i == j:
                for k in range(i-1, i+2):
                    if 1 > k or k >= n+1:
                        continue
                    # s += w[i-1] * w[j-1] * w[k-1] * hat(x, i, n) * hat(x, j, n) * hat(x, k, n)
                    s += w[i-1] * w[j-1] * w[k-1] * h[i-1] * h[j-1] * h[k-1]
            else:
                # for k in (i, j)
                s += w[i-1]**2. * w[j-1] * h[i-1]**2. * h[j-1]
                s += w[j-1]**2. * w[i-1] * h[j-1]**2. * h[i-1]

    return s


# cdef double _dot_pow(double[:] a, double[:] b, size_t pow, size_t[:] seen_, size_t depth) except *:
#
#     # If the previous iterates are equal, we iter around them, otherwise
#     #  we iter through the two options.
#     seen = list(set(seen_))
#
#     cdef:
#         double s = 0
#         double p = 1
#
#     if len(s) == 1:
#         for i in range(s[0]-1, s[0]+2):
#
#
#             if depth == pow:
#                 p = 1
#                 for j in seen_:
#

# <Hat Integrals>

# # TODO this is the basis for a recursive version. We just need ijk to be in a collection.
# cdef double _int_hat_dot_pow3_hat_inner(i, j, k, double[:] w, size_t n, double[:] gw) except *:
#
#     cdef size_t ct
#
#     if i == j and i == k:
#         for m in range(i-1, i+2):
#             ct = idx_ct_4hats(i, j, k, m, n, True)
#             gw[m] += w[i] * w[j] * w[k] *
#     else:
#         for m in (i, j, k):
#             pass
#
#
# cpdef void int_hat_dot_pow3_hat(double[:] w, size_t n, double[:] gw) except *:
#     cdef:
#         size_t i, j, k, m
#
#     gw[:] = 0
#
#     for i in range(1, n+1):
#         for j in range(i-1, i+2):
#             if 1 > j or j >= n+1:
#                 continue
#             if i == j:
#                 for k in range(i-1, i+2):
#                     if 1 > k or k >= n+1:
#                         continue
#                     _int_hat_dot_pow3_hat_inner(i, j, k, w, n, gw)
#             else:
#                 for k in (i, j):
#                     _int_hat_dot_pow3_hat_inner(i, j, k, w, n, gw)

# TODO implement the fancy recursive version commented out above.
cpdef double int_hat_dot_pow3_hat(double[:] w, size_t n, double[:] gw) except *:

    cdef:
        size_t i, j, k, m, ct
        size_t[:] s

    for m in range(1, n+1):
        for i in range(m-1, m+2):
            for j in range(m-1, m+2):
                for k in range(m-1, m+2):
                    s = np.array(list({m, i, j, k}), dtype=np.uintp)
                    if len(s) > 2 or (len(s) == 2 and abs(s[0] - s[1]) > 1):
                        continue
                    # Otherwise...compute.
                    ct = idx_ct_4hats(i, j, k, m, n, True)
                    gw[m] += w[i] * w[j] * w[k] * int_4hats(ct, n)


cpdef double int_hatp_dot_pow3_hatp(double[:] w, size_t n, double[:] gw) except *:

    cdef:
        size_t i, j, k, m, ct
        size_t[:] s

    for m in range(1, n+1):
        for i in range(m-1, m+2):
            for j in range(m-1, m+2):
                for k in range(m-1, m+2):
                    s = np.array(list({m, i, j, k}), dtype=np.uintp)
                    if len(s) > 2 or (len(s) == 2 and abs(s[0] - s[1]) > 1):
                        continue
                    # Otherwise...compute.
                    ct = idx_ct_4hats(i, j, k, m, n, True)
                    gw[m] += w[i] * w[j] * w[k] * int_4hatsp(ct, n)


cpdef double int_hatp_dot_pow1_hatp(double[:] w, size_t n, double[:] gw) except *:

    cdef:
        size_t i, m, ct

    for m in range(1, n+1):
        for i in range(m-1, m+2):
            if m == i: ct = 1
            if m != i: ct = 2
            gw[m] += w[i] * int_2hatsp(ct, n)


cpdef size_t idx_ct_4hats(size_t i, size_t j, size_t k, size_t m, size_t n, bint check) except *:

    if check:
        s = list({i, j, k, m})
        if len(s) > 2 or (len(s) == 2 and abs(s[0] - s[1]) > 1):
            raise ValueError("Integral would evaluate to zero: hats not all adjacent.")

    # i is counted in ct1
    cdef size_t ct1=1, ct2=0

    if j == i:
        ct1 += 1
    else:
        ct2 += 1

    if k == i:
        ct1 += 1
    else:
        ct2 += 1

    if m == i:
        ct1 += 1
    else:
        ct2 += 1

    return max(ct1, ct2)


cdef double int_4hats(size_t idx_ct, size_t n_) except *:
    cdef double n = <double>n_

    if idx_ct == 2:
        return 17./(10.*(n+1.))
    if idx_ct == 3:
        return 11./(20.*(n+1.))
    if idx_ct == 4:
        return 2./(5.*(n+1.))


cdef double int_4hatsp(size_t idx_ct, size_t n_) except *:
    cdef double n = <double>n_

    if idx_ct == 2:
        return (n+1.)**3.
    if idx_ct == 3:
        return -(n+1.)**3.
    if idx_ct == 4:
        return 2*(n+1.)**3.

cdef double int_2hatsp(size_t idx_ct, size_t n_) except *:
    cdef double n = <double>n_

    if idx_ct == 1:
        return -(n+1.)
    if idx_ct == 2:
        return 2.*(n+1.)

# <Hat Integrals>




