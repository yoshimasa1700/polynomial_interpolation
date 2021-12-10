import numpy as np
import matplotlib.pyplot as plt


def open_uniform_knot_vector(m, n):
    u = np.zeros(m)

    for i in range(m):
        if i < n + 1:
            u[i] = 0
            continue

        if i > m - (n + 1):
            u[i] = m - 1 - 2 * n
            continue

        u[i] = i - n

    return u


def basis_func(u, j, k, t):
    w1 = 0.
    w2 = 0.

    if k == 0:
        if t > u[j] and t <= u[j + 1]:
            return 1.
        else:
            return 0.

    if (u[j+k+1] - u[j+1]) == 0:
        w1 = 0.
    else:
        w1 = (u[j+k+1] - t) / (u[j+k+1] - u[j+1])

    if u[j+k] - u[j] == 0:
        w2 = 0.0
    else:
        w2 = (t - u[j]) / (u[j+k] - u[j])

    return w1 * basis_func(u,j+1,k-1,t) + w2 * basis_func(u,j,k-1,t)


P = np.array(
    [
        [0, 0],
        [1, 2],
        [3, 2],
        [3, 0],
        [5, 2],
        [6, 0]
    ], dtype=float)

p = P.shape[0] # number of control points
n = 3 # degree of b spline
m = p + n + 1 # number of knot in knot vector

u = open_uniform_knot_vector(m, n)

t = np.arange(0, u[-1], 0.01)

# calc b spline
S = np.zeros((t.shape[0], 2))

S[0] = P[0]

for i in range(2, t.shape[0]):
    for j in range(p):
        b = basis_func(u, j, n, t[i])
        S[i] = S[i] + P[j] * b

plt.plot(S[:, 0], S[:, 1], label="B-spline")
plt.plot(P[:, 0], P[:, 1], label="Control point", marker='.', markersize=15, linestyle='None')
plt.legend()
plt.grid()

plt.show()
