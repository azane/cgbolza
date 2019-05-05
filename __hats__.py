import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def onehat(x, u):
    if x < u - 1:
        return 0
    if x > u + 1:
        return 0
    if x < u:
        return x - u + 1
    if x > u:
        return -x + u + 1


def onehat_vec(x, u):
    z1 = x < u - 1
    z2 = x > u + 1
    z = ~np.logical_or(z1, z2)

    xl = x < u
    xr = x > u

    return ((xl * (x - u + 1)) + (xr * (-x + u + 1)))*z


def onehat_vec2(x, i, n):
    z1 = x < (i-1)/(n+1)
    z2 = x > (i+1)/(n+1)
    z = ~np.logical_or(z1, z2)

    xl = x < i/(n+1)
    xr = x > i/(n+1)

    xlv = (n + 1)*x - i + 1
    xrv = -(n + 1)*x + i + 1

    return xl * xlv * z + xr * xrv * z


if __name__ == "__main__":

    # y = []
    # p = np.random.rand(3)*2.-1
    # for x in np.linspace(-2, 2):
    #     yy = np.sum([pp*onehat(x, u) for (pp, u) in zip(p, [-1, 0, 1])])
    #     y.append(yy)
    # plt.plot(y)
    # plt.show()

    # nn = [4]
    # bb = (0, 1)
    # xx = np.linspace(*bb, 1000)
    # for n in nn:
    #     uu = np.arange(n)+1
    #     pp = np.random.normal(size=(len(uu),))
    #     f = []
    #     for u in uu:
    #         f.append(onehat_vec((n+1)*xx, u))
    #     plt.figure()
    #     fa = np.array(f)
    #     plt.plot(fa.T)
    #     plt.plot((pp[:, None] * fa).sum(axis=0), color='black')
    # plt.show()

    nn = [4, 10]
    bb = (0, 1)
    xx = np.linspace(*bb, 1000)
    for n in nn:
        pp = np.random.normal(size=(n,))
        f = []
        for i in range(1, n+1):
            f.append(onehat_vec2(xx, i, n))
        plt.figure()
        fa = np.array(f)
        plt.plot(fa.T)
        plt.plot((pp[:, None] * fa).sum(axis=0), color='black')
    plt.show()