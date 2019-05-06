from scipy.optimize import fmin_cg
from src.bolza import hhp
import numpy as np
import matplotlib.pyplot as plt
from src.cg import cg


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
    n = 30
    xx = np.linspace(0, 1, 1000)
    h, hp = hhp(xx, n)
    w = np.random.normal(size=(n,))

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
    ucolor = 'tab:blue'

    fax = uax.twinx()
    fcolor = 'tab:orange'

    err = []
    gnorm = []

    def callback(w_):

        uax.clear()
        uax.set_xlabel('x')
        uax.set_ylabel('u(x)', color=ucolor)
        uax.tick_params(axis='y', labelcolor=ucolor)
        uax.plot(xx, uu(w_, h), color=ucolor)

        fax.clear()
        fax.set_ylabel("f(u', u)", color=fcolor)
        fax.tick_params(axis='y', labelcolor=fcolor)
        fax.plot(xx, f(w_, h, hp), color=fcolor)

        err.append(If(w_, h, hp))
        gnorm.append(np.linalg.norm(Ifp(w_, h, hp)))

        plt.pause(0.01)


    # fmin_cg(f=lambda w_: np.log(If(w_, h, hp)),
    #         fprime=lambda w_: Ifp(w_, h, hp)/If(w_, h, hp),
    #         x0=w,
    #         disp=True,
    #         callback=callback,
    #         maxiter=200,
    #         norm=2)
    cg(f=lambda w_: np.log(If(w_, h, hp)),
       fp=lambda w_: Ifp(w_, h, hp)/If(w_, h, hp),
       x0=w,
       callback=callback,
       maxiter=500)

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


