from src.hats import hat
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    xx = np.linspace(0, 1, 1000)
    yy = []
    n = 10
    for i in range(1, n+1):
        yy.append([])
        for x in xx:
            yy[-1].append(hat(x, i, n))

    fa = np.array(yy)
    plt.plot(fa.T)
    plt.show()