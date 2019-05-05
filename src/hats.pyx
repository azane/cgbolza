

cpdef say_hello_to(name):
    print("Helloo %s!" % name)


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
                s += w[i-1]**2 * w[j-1] * hat(x, i, n)**2 * hat(x, j, n)
                s += w[j-1]**2 * w[i-1] * hat(x, j, n)**2 * hat(x, i, n)

    return s