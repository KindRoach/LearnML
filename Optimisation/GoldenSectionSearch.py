from math import e, log
import numpy as np
import matplotlib.pyplot as plt

LOWER = 1
UPPER = 2
INC = 1e-3


def f(x: float) -> float:
    a = 8 * e ** (1 - x)
    b = 7 * log(x)
    return a + b


def draw_graph(f):
    x = np.arange(start=LOWER, stop=UPPER + INC, step=INC)
    y = [f(xi) for xi in x]
    plt.plot(x, y)
    plt.show()


def gss(f):
    p = (1 - 0.61803)
    gap = p * (UPPER - LOWER)
    a0, a1 = LOWER, LOWER + gap
    b0, b1 = UPPER, UPPER - gap
    f_a, f_b = f(a1), f(b1)

    res = [["a", "b", "f(a)", "f(b)", "new uncertainty interval"]]
    while (b0 - a0) > 0.23:
        res.append([a1, b1, f_a, f_b])
        if f_a < f_b:
            res[-1].append([a0, b1])
            b0, b1 = b1, a1
            a1 = a0 + p * (b0 - a0)
            f_b, f_a = f_a, f(a1)
        else:
            res[-1].append([a1, b0])
            a0, a1 = a1, b1
            b1 = b0 - p * (b0 - a0)
            f_a, f_b = f_b, f(b1)

    return res


draw_graph(f)
print("\n".join(str(x) for x in gss(f)))
