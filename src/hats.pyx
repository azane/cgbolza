

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


cpdef double hat_dot_pow3(double x, double[:] w, size_t n) except *:
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
                    s += w[i-1] * w[j-1] * w[k-1] * hat(x, i, n) * hat(x, j, n) * hat(x, k, n)
            else:
                # for k in (i, j)
                s += w[i-1]**2. * w[j-1] * hat(x, i, n)**2. * hat(x, j, n)
                s += w[j-1]**2. * w[i-1] * hat(x, j, n)**2. * hat(x, i, n)

    return s

# TODO change dot pow to just take vectors.
# TODO precompute the hat vectors for each value, this will actually be faster.
# TODO also this will allow us to pass in gradient values or actual values.
# TODO make a dot_pow4 so we can easily compute the the full integrand.


# <Hat Integrals>
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


cdef double int_4hats(size_t idx_ct, size_t n_):
    cdef double n = <double>n_

    if idx_ct == 2:
        return 17./(10.*(n+1.))
    if idx_ct == 3:
        return 11./(20.*(n+1.))
    if idx_ct == 4:
        return 2./(5.*(n+1.))


cdef double int_4hatsp(size_t idx_ct, size_t n_):
    cdef double n = <double>n_

    if idx_ct == 2:
        return (n+1.)**3.
    if idx_ct == 3:
        return -(n+1.)**3.
    if idx_ct == 4:
        return 2*(n+1.)**3.

cdef double int_2hatsp(size_t idx_ct, size_t n_):
    cdef double n = <double>n_

    if idx_ct == 1:
        return -(n+1.)
    if idx_ct == 2:
        return 2.*(n+1.)

# <Hat Integrals>




