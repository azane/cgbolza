import matplotlib.pyplot as plt
import numpy as np
from src.hats import \
    int_hat_dot_pow3_hat,\
    int_hatp_dot_pow3_hatp,\
    int_hatp_dot_pow1_hatp,\
    hat_dot_pow3,\
    hatp_dot_pow3,\
    hat, hatp


def f1(xx, w, n):
    yy = []
    for x in xx:
        h = np.array([hat(x, i, n) for i in range(1, n + 1)])
        yy.append(hat_dot_pow3(x, w, n)*h)
    return np.array(yy)


def f2(xx, w, n):
    yy = []
    for x in xx:
        hp = np.array([hatp(x, i, n) for i in range(1, n+1)])
        yy.append(w.dot(hp)*hp)
    return np.array(yy)


def f3(xx, w, n):
    yy = []
    for x in xx:
        hp = np.array([hatp(x, i, n) for i in range(1, n + 1)])
        yy.append(hatp_dot_pow3(x, w, n)*hp)
    return np.array(yy)


if __name__ == "__main__":
    xx = np.linspace(0, 1, 10000)
    n = 5
    w = np.random.normal(size=(n,))

    f1y = f1(xx, w, n)
    f2y = f2(xx, w, n)
    f3y = f3(xx, w, n)

    print("Approximations:")
    print("gw1:", f1y.sum(0)/len(xx))
    print("gw2:", f2y.sum(0)/len(xx))
    print("gw3:", f3y.sum(0)/len(xx))

    plt.figure()
    plt.plot(f1y)
    plt.figure()
    plt.plot(f2y)
    plt.figure()
    plt.plot(f3y)
    plt.figure()
    plt.show()

    # TODO this definitely does not work...lotso debugs to do here. : /
    print("Exact:")

    gw1 = np.empty(n, dtype=float)
    int_hat_dot_pow3_hat(w, n, gw1)
    print("gw1:", gw1)

    gw2 = np.empty(n, dtype=float)
    int_hatp_dot_pow1_hatp(w, n, gw2)
    print("gw2:", gw2)

    gw3 = np.empty(n, dtype=float)
    int_hatp_dot_pow3_hatp(w, n, gw3)
    print("gw3:", gw3)
