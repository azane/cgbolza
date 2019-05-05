

cpdef say_hello_to(name):
    print("Helloo %s!" % name)


cpdef double hat(double x, size_t i_, size_t n_) except *:
    cdef:
        double i = <double>i_
        double np1 = <double>(n_+1)

    if x < ((i-1)/np1): return 0
    if x > ((i+1)/np1): return 0

    if x < (i/np1):
        return np1*x - i + 1
    if x > (i/np1):
        return i - np1*x + 1


cpdef double hat_dot_pow3(double x, double[:] w, size_t n) except *:
    cdef:
        double s = 0
        size_t i, j, k

    for i in range(n):
        for j in range(i-1, i+2):
            if 0 > j or j >= n:
                continue
            if i == j:
                for k in range(i-1, i+2):
                    if 0 > k or k >= n:
                        continue
                    s += w[i] * w[j] * w[k] * hat(x, i, n) * hat(x, j, n) * hat(x, k, n)
            else:
                # for k in (i, j)
                s += w[i]**2 * w[j] * hat(x, i, n)**2 * hat(x, j, n)
                s += w[j]**2 * w[i] * hat(x, j, n)**2 * hat(x, i, n)

    return s