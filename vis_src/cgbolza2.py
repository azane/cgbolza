from scipy.optimize import fmin_cg
from src.bolza import hhp
import numpy as np
import matplotlib.pyplot as plt
from src.cg import cg
from cgbolza_h import *

if __name__ == "__main__":

    errs = []
    xx = np.linspace(0, 1, 500)

    errf, errax = plt.subplots()
    errf.suptitle("Error Over Different N")
    errax.set_yscale('log')
    errax.set_ylabel('Log Functional Value')
    errax.set_xlabel('Iterations')

    def callback(w_, h_, hp_):
        errs[-1].append(If(w_, h_, hp_))

    for n in  [10, 30, 50, 100, 200, 300, 400]: # [10, 30, 50, 100, 300]:
        h, hp = hhp(xx, n)
        w = np.empty((n,))
        w[::2] = 1
        w[1::2] = -1
        errs.append([])

        r = fmin_cg(f=lambda w_: np.log(If(w_, h, hp) + 1),
                fprime=lambda w_: Ifp(w_, h, hp) / (If(w_, h, hp) + 1),
                x0=w,
                disp=True,
                callback=lambda w_: callback(w_, h, hp),
                maxiter=1000,
                norm=2)

        wstar = r

        # <Plot Result>
        fig, uax = plt.subplots()
        fig.suptitle(f"n = {n}")
        fig.subplots_adjust(right=0.75)
        ucolor = 'tab:blue'

        fax = uax.twinx()
        fcolor = 'tab:orange'

        pax = uax.twinx()
        pcolor = 'tab:green'

        err = []
        gnorm = []
        percentxivoff = []
        WW = []

        uax.set_xlabel('x')
        uax.set_ylabel('u(x)', color=ucolor)
        uax.tick_params(axis='y', labelcolor=ucolor)
        uax.plot(xx, uu(wstar, h), color=ucolor)

        pax.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(pax)
        pax.spines["right"].set_visible(True)
        pax.set_ylabel("u'(x)", color=pcolor)
        pax.tick_params(axis='y', labelcolor=pcolor)
        pax.plot(xx, np.fabs(upup(wstar, hp)), color=pcolor, linestyle=":")

        fax.set_ylabel("f(u', u)", color=fcolor)
        fax.tick_params(axis='y', labelcolor=fcolor)
        fxiv = (upup(wstar, hp) ** 2 - 1) ** 2
        fv = f(wstar, h, hp)
        fax.plot(xx, fv, color=fcolor)
        # </Plot Result>

        errax.plot(errs[-1], label=f"n = {n}")

    errax.legend()

    plt.show()
