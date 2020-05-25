import math

from scipy.optimize import minimize


def f(x1: float, x2: float) -> float:
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def gradient(x1: float, x2: float) -> (float, float):
    g1 = 400 * x1 * (x1 ** 2 - x2) + 2 * x1 - 2
    g2 = 200 * (x2 - x1 ** 2)
    return g1, g2


def normal(x1: float, x2: float) -> float:
    return math.sqrt(x1 ** 2 + x2 ** 2)


if __name__ == '__main__':
    x1, x2 = -2., -2.
    g_normal = float("inf")
    while g_normal >= 1e-4:
        g1, g2 = gradient(x1, x2)


        def f_a(a: float) -> float:
            return f(x1 - a * g1, x2 - a * g2)


        alpha = minimize(f_a, [0.1]).x[0]
        x1 = x1 - alpha * g1
        x2 = x2 - alpha * g2
        g_normal = normal(g1, g2)

    print(x1, x2)
