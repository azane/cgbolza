import sympy as sp
import sympy.plotting as plt
import numpy as np
import time
from src.hello import say_hello_to


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


def ghats_x_sp(x, n):

    hats = []
    for i in range(1, n+1):
        hi = sp.Piecewise(
            (0, x < (i-1)/(n+1)),
            (0, x > (i+1)/(n+1)),
            ((n + 1), x < i/(n+1)),
            (-(n + 1), x > i/(n+1))
        )
        hats.append(hi)

    return hats


def dot(a, b):
    assert len(a) == len(b)
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def pow3_dot_refine(a, b):

    s = 0
    n = len(a)
    for i in range(n):
        for j in (i-1, i, i+1):
            if 0 > j or j >= n:
                continue
            if i == j:
                for k in (i-1, i, i+1):
                    if 0 > k or k >= n:
                        continue
                    s += a[i] * a[j] * a[k] * b[i] * b[j] * b[k]
            else:
                for k in (i, j):
                    s += a[i] * a[j] * a[k] * b[i] * b[j] * b[k]

    return s

if __name__ == "__main__":
    t0 = time.time()
    x = sp.symbols('x')
    n = 10
    hats = hats_sp(x, n)

    p = plt.plot(hats[0], (x, 0, 1), show=False)
    # for h in hats[1:]:
    #     p.append(plt.plot(h, (x, 0, 1), show=False)[0])
    # p.show()

    w = sp.symbols(', '.join([f'w{i+1}' for i in range(n)]))

    ghats_x = [h.diff(x) for h in hats]

    u = dot(w, hats)
    gu_x = dot(w, ghats_x)

    print(time.time() - t0)

    gu_x3 = pow3_dot_refine(w, ghats_x)
    u3 = pow3_dot_refine(w, hats)

    print(time.time() - t0)

    f = (gu_x**2 - 1)**2 + u**4
    gf_w = []
    for i in range(n):
        gf_wi = 4*(ghats_x[i]*gu_x3 - ghats_x[i]*gu_x + hats[i]*u3)
        gf_w.append(gf_wi)
        print(gf_wi)

    print(time.time() - t0)

    # gI_w = []
    # for gf_wi in gf_w:
    #     gI_wi = sp.integrate(gf_wi, (x, 0, 1), manual=True)
    #     print(time.time() - t0)
    #     gI_w.append(gI_wi)

