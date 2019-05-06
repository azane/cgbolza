def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


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