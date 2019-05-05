from src.hats import hat, hat_dot_pow3
import numpy as np


def test_hat_dot_pow3():

    xx = np.linspace(0, 1, 1000)
    n = 10
    w = np.random.normal(size=(10,))
    for x in xx:
        h = []
        for i in range(1, n+1):
            h.append(hat(x, i, n))
        h = np.array(h)

        assert np.isclose(w.dot(h)**3, hat_dot_pow3(x, w, n))
