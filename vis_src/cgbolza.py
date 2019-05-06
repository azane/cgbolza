from scipy.optimize import fmin_cg
from src.bolza import hhp
import numpy as np
import matplotlib.pyplot as plt
from src.cg import cg
from cgbolza_h import *


if __name__ == "__main__":
    n = 100
    xx = np.linspace(0, 1, 600)
    h, hp = hhp(xx, n)
    # w = np.random.normal(loc=1., scale=0.00, size=(n,))
    w = np.empty((n,))
    w[::2] = 1
    w[1::2] = -1

    if n == 2 and len(xx) <= 500:
        delta = 0.007
        w1 = np.arange(-1., 1., delta)
        w2 = np.arange(-1., 1., delta)
        W1, W2 = np.meshgrid(w1, w2)

        W = np.vstack((W1[None, ...], W2[None, ...])).transpose(1, 2, 0)[:, :, None, :]
        u = (W * h[None, None, ...]).sum(-1)
        up = (W * h[None, None, ...]).sum(-1)
        F = np.log(((up ** 2 - 1) ** 2 + u ** 4).sum(-1))

        _, ax = plt.subplots()
        CS = ax.contourf(W1, W2, F)
        plt.colorbar(CS)
        ax.set_title('Contours of Functional, 2 Parameters')
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        plt.show()

        exit()

    # <Contours for u and u'>
    delta = 0.005
    U_ = np.arange(-1.5, 1.5, delta)
    XI_ = np.arange(-1.5, 1.5, delta)
    U, XI = np.meshgrid(U_, XI_)
    F = np.log(((XI ** 2 - 1) ** 2 + U ** 4) + 1)
    _, ax = plt.subplots()
    CS = ax.contourf(U, XI, F)
    plt.colorbar(CS)
    ax.set_title('Log Contours of Functional')
    ax.set_xlabel('u')
    ax.set_ylabel('xi')
    # <Contours for u and u'>

    # plt.figure()
    # plt.plot(h)

    # plt.plot(uu(w, h), color='black')
    #
    # plt.figure()
    # plt.plot(hp)
    # plt.plot(upup(w, hp), color='black')
    #
    # plt.figure()
    # fv = f(w, h, hp)
    # plt.plot(fv)
    #
    # plt.figure()
    # fpv = fp(w, h, hp)
    # plt.plot(fpv)
    #
    # plt.show()

    plt.ion()
    fig, uax = plt.subplots()
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

    def callback(w_):

        uax.clear()
        uax.set_xlabel('x')
        uax.set_ylabel('u(x)', color=ucolor)
        uax.tick_params(axis='y', labelcolor=ucolor)
        uax.plot(xx, uu(w_, h), color=ucolor)

        pax.clear()
        pax.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(pax)
        pax.spines["right"].set_visible(True)
        pax.set_ylabel("u'(x)", color=pcolor)
        pax.tick_params(axis='y', labelcolor=pcolor)
        pax.plot(xx, np.fabs(upup(w_, hp)), color=pcolor, linestyle=":")

        fax.clear()
        fax.set_ylabel("f(u', u)", color=fcolor)
        fax.tick_params(axis='y', labelcolor=fcolor)
        fxiv = (upup(w_, hp)**2 - 1)**2
        fv = f(w_, h, hp)
        fax.plot(xx, fv, color=fcolor)
        # fax.plot(xx, fxiv, color='tab:orange', label="f(u', 0)")
        # fax.plot(xx, fuv, color='tab:brown', label="f(0, u)")
        # fax.legend()

        err.append(If(w_, h, hp))
        gnorm.append(np.linalg.norm(Ifp(w_, h, hp)))
        percentxivoff.append(fxiv.sum()/fv.sum())
        WW.append(w_)

        plt.pause(0.01)

    ret = None
    fmin_cg(f=lambda w_: np.log(If(w_, h, hp) + 1),
            fprime=lambda w_: Ifp(w_, h, hp)/(If(w_, h, hp) + 1),
            x0=w,
            disp=True,
            callback=callback,
            maxiter=250,
            norm=2)
    # ret = cg(f=lambda w_: np.log(If(w_, h, hp) + 1),
    #          fp=lambda w_: Ifp(w_, h, hp)/(If(w_, h, hp) + 1),
    #          x0=w,
    #          callback=callback,
    #          maxiter=250)
    # ret = cg(f=lambda w_: If(w_, h, hp),
    #          fp=lambda w_: Ifp(w_, h, hp),
    #          x0=w,
    #          callback=callback,
    #          maxiter=250)

    plt.ioff()

    if ret is not None:
        f = plt.figure()
        f.suptitle("Normalized Search Direction Dot Products")
        plt.xlabel("Iterations")
        plt.ylabel("Iterations")
        im = plt.imshow(ret.dot(ret.T))
        f.colorbar(im)

    plt.figure()
    plt.ylabel("Log Functional")
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.plot(err)

    plt.figure()
    plt.ylabel("Log Gradient Norm")
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.plot(gnorm)

    f = plt.figure()
    f.suptitle("Portion of Error:\nIf(u', 0) / If(u', u)")
    plt.xlabel("Iterations")
    plt.plot(percentxivoff)

    f = plt.figure()
    f.suptitle("Parameters")
    plt.xlabel("Iterations")
    plt.plot(np.array(WW))

    plt.show()


