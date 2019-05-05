from src.hats import hat, hat_dot_pow3, idx_ct_4hats
import numpy as np
import pytest


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


def test_idx_ct_4hats():

    n = 10

    with pytest.raises(ValueError):
        i, j, k, m = 1, 1, 1, 3
        idx_ct_4hats(i, j, k, m, n, True)

    i, j, k, m = 1, 1, 2, 2
    print({i, j, k, m})
    assert idx_ct_4hats(i, j, k, m, n, True) == 2

    i, j, k, m = 2, 1, 2, 1
    assert idx_ct_4hats(i, j, k, m, n, True) == 2

    i, j, k, m = 1, 1, 1, 1
    assert idx_ct_4hats(i, j, k, m, n, True) == 4

    i, j, k, m = 4, 5, 5, 4
    assert idx_ct_4hats(i, j, k, m, n, True) == 2

    i, j, k, m = 5, 4, 4, 4
    assert idx_ct_4hats(i, j, k, m, n, True) == 3

    i, j, k, m = 6, 6, 7, 6
    assert idx_ct_4hats(i, j, k, m, n, True) == 3

    i, j, k, m = 7, 7, 7, 6
    assert idx_ct_4hats(i, j, k, m, n, True) == 3

    with pytest.raises(ValueError):
        i, j, k, m = 3, 4, 5, 4
        idx_ct_4hats(i, j, k, m, n, True)
