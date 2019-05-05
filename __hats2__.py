import sympy as sp
import sympy.plotting as plt
import numpy as np
import time


def onehat_vec2(x, i, n):
    z1 = x < (i-1)/(n+1)
    z2 = x > (i+1)/(n+1)
    z = ~np.logical_or(z1, z2)

    xl = x < i/(n+1)
    xr = x > i/(n+1)

    xlv = (n + 1)*x - i + 1
    xrv = -(n + 1)*x + i + 1

    return xl * xlv * z + xr * xrv * z


def hats_sp(x, n):

    hats = []
    for i in range(1, n+1):
        hi = sp.Piecewise(
            (0, x < (i-1)/(n+1)),
            (0, x > (i+1)/(n+1)),
            ((n + 1)*x - i + 1, x < i/(n+1)),
            (-(n + 1)*x + i + 1, x > i/(n+1))
        )
        hats.append(hi)

    return hats


if __name__ == "__main__":
    t = time.time()
    x = sp.symbols('x')
    hats = hats_sp(x, 10)

    p = plt.plot(hats[0], (x, 0, 1), show=False)
    for h in hats[1:]:
        p.append(plt.plot(h, (x, 0, 1), show=False)[0])
    p.show()
