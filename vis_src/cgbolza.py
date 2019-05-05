from scipy.optimize import fmin_cg
from src.bolza import hhp
import numpy as np
import matplotlib.pyplot as plt


def uu(w_, h_):
    return (w_[None, :] * h_).sum(1, keepdims=True)  # (x, 1)


def upup(w_, hp_):
    return (w_[None, :] * hp_).sum(axis=1, keepdims=True)  # (x, 1)


def f(w_, h_, hp_):

    up = upup(w_, hp_)
    u = uu(w_, h_)

    return (up**2 - 1)**2 + u**4


def fp(w_, h_, hp_):

    # h_, hp_  # .shape == (x, n)

    up = upup(w_, hp_)  # (x, 1)
    u = uu(w_, h_)  # (x, 1)

    return 4. * ((up**3 - up) * hp_ + u**3 * h_)


def If(w_, h_, hp_):
    return f(w_, h_, hp_).sum()


def Ifp(w_, h_, hp_):
    return fp(w_, h_, hp_).sum(axis=0)


if __name__ == "__main__":
    n = 40
    xx = np.linspace(0, 1, 3000)
    h, hp = hhp(xx, n)
    w = np.random.normal(size=(n,))

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
    plt.figure()
    plt.xlabel("x")

    err = []
    gnorm = []

    def callback(w_):
        plt.clf()
        u = uu(w_, h)
        plt.plot(u, label="u(x)")
        plt.plot(np.log(f(w_, h, hp) + 1), label="log(f(u', u))")
        plt.legend()
        plt.pause(0.01)
        err.append(If(w_, h, hp))
        gnorm.append(np.linalg.norm(Ifp(w_, h, hp)))


    fmin_cg(f=lambda w_: np.log(If(w_, h, hp)),
            fprime=lambda w_: Ifp(w_, h, hp)/If(w_, h, hp),
            x0=w,
            disp=True,
            callback=callback,
            gtol=0.1,
            maxiter=120,
            norm=2)

    plt.ioff()

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
    plt.show()
