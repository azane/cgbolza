import sympy as sp


def hat(x, i, n):
    hi = sp.Piecewise(
        (0, x < (i - 1) / (n + 1)),
        (0, x > (i + 1) / (n + 1)),
        ((n + 1) * x - i + 1, x < i / (n + 1)),
        (-(n + 1) * x + i + 1, x > i / (n + 1))
    )
    return hi


def hats_sp(x, n):
    hats = []
    for i in range(1, n + 1):
        hats.append(hat(x, n, i))

    return hats


if __name__ == "__main__":
    x, i, j, n, p = sp.symbols('x, i, j, n, p')
    rsv = i - (n+1)*x + 1
    lsv = (n+1)*x - i + 2

    bi = (i / (n + 1), (i + 1) / (n + 1))
    p22 = sp.integrate(rsv ** 2 * lsv ** 2, (x, *bi))
    p31 = sp.integrate(rsv ** 3 * lsv ** 1, (x, *bi))
    p40 = 2 * sp.integrate(rsv ** 4, (x, *bi))
    p22 = sp.simplify(p22)
    p31 = sp.simplify(p31)
    p40 = sp.simplify(p40)

    print("22:", p22)
    print("31:", p31)
    print("40:", p40)

    rsvp = rsv.diff(x)
    lsvp = lsv.diff(x)

    p22p = sp.integrate(rsvp ** 2 * lsvp ** 2, (x, *bi))
    p31p = sp.integrate(rsvp ** 3 * lsvp ** 1, (x, *bi))
    p13p = sp.integrate(rsvp ** 1 * lsvp ** 3, (x, *bi))
    p40p = 2 * sp.integrate(rsvp ** 4, (x, *bi))
    p11p = sp.integrate(rsvp ** 1 * lsvp ** 1, (x, *bi))
    p20p = 2 * sp.integrate(rsvp ** 2, (x, *bi))
    p22p = sp.simplify(p22p)
    p31p = sp.simplify(p31p)
    p13p = sp.simplify(p13p)
    p40p = sp.simplify(p40p)
    p11p = sp.simplify(p11p)
    p20p = sp.simplify(p20p)

    print("22p:", p22p)
    print("31p:", p31p)
    print("13p:", p31p)
    print("40p:", p40p)
    print("11p:", p11p)
    print("20p:", p20p)





