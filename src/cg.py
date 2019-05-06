from typing import *
import numpy as np
from scipy.optimize import line_search

# https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method


def cg(f: Callable, fp: Callable, x0: np.ndarray, callback: Callable, maxiter: int):

    beta = 0
    alpha = 0.1

    # Last f value.
    fl = f(x0)

    # Current position.
    xn = np.copy(x0)

    # Current descent direction.
    sn = xn*0

    # Last steepest descent direction.
    gxl = xn*0

    dirs = []

    for n in range(maxiter):

        # Calculate steepest descent direction.
        gxn = -fp(xn)

        # Compute beta. Polak Ribiere.
        if n > 0:
            den = gxl.dot(gxl)
            if np.isclose(0, den):
                pass
            else:
                beta = gxn.dot(gxn - gxl)/den

        # Update conjugate direction.
        sn = gxn + beta * sn

        snn = np.linalg.norm(sn)
        if np.isclose(0, snn):
            print("Converged!")
            break

        # Line Search HACK
        res = line_search(f=f, myfprime=fp, xk=xn, pk=sn, gfk=-gxn)
        nalpha = res[0]
        if nalpha is None:
            dirs.append(sn * np.nan)
            continue

        dirs.append(sn / snn)

        alpha = nalpha
        xl = xn
        xn = xn + alpha * sn

        callback(xn)

        # if np.linalg.norm(xn - xl) < 1e-3:
        #     print("Converged!")
        #     break

        # Line search.
        # HACK instead for test.
        # xn = xn + 0.05 * sn / np.linalg.norm(sn)
        # callback(xn)
        # last_good_alpha = 0
        # while True:
        #     fn = f(xn + alpha*sn)
        #
        #     callback(xn + alpha * sn)
        #
        #     if abs(fn - fl) < 1e-2:
        #         fl = fn
        #         break
        #
        #     # If better, keep going.
        #     if fn < fl:
        #         last_good_alpha = alpha
        #         alpha = alpha * 1.1
        #     # If worse, then halve the step from the last good alpha.
        #     elif fn > fl:
        #         alpha = last_good_alpha + abs(last_good_alpha - alpha) / 2
        #
        #     fl = fn

        gxl = gxn

    return np.array(dirs)