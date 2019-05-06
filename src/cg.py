from typing import *
import numpy as np

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

    for n in range(maxiter):

        # Calculate steepest descent direction.
        gxn = -fp(xn)

        # Compute beta. Polak Ribiere.
        if n > 0:
            beta = gxn.dot(gxn - gxl)/gxl.dot(gxl)

        # Update conjugate direction.
        sn = gxn + beta * sn

        # Line search.
        last_good_alpha = 0
        while True:
            fn = f(xn + alpha*sn)

            callback(xn + alpha * sn)

            if abs(fn - fl) < 1e-2:
                fl = fn
                break

            # If better, keep going.
            if fn < fl:
                last_good_alpha = alpha
                alpha = alpha * 1.1
            # If worse, then halve the step from the last good alpha.
            elif fn > fl:
                alpha = last_good_alpha + abs(last_good_alpha - alpha) / 2

            fl = fn

        gxl = gxn