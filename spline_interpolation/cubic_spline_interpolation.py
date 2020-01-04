import numpy as np
import matplotlib.pyplot as plt

# sample points.
sample_points = [(-4, -128),
                 (-2, -16),
                 (0, 0),
                 (2, -40),
                 (4, 16),
                 (7, 51)]

n = len(sample_points)

dimention = 3  # cubic

sample_points_lists = [list(x) for x in sample_points]
sample_points_np = np.array(sample_points_lists, dtype=np.float32)

# calc intermidiate values.
h = sample_points_np[1:, 0] - sample_points_np[0: -1, 0]
b = np.divide(sample_points_np[1:, 1] - sample_points_np[0: -1, 1], h)


# create matrix A.
shape = (n - 2, n - 1)
A = np.zeros(shape, dtype=np.float32)

for idx in xrange(n - 2):
    # fill diagonal components.
    A[idx][idx] = 2 * (h[idx] + h[idx + 1])

    # fill last components.
    A[idx][n - 2] = 6 * (b[idx + 1] - b[idx])

for idx in xrange(0, n - 3):
    A[idx + 1][idx] = h[idx + 1]
    A[idx][idx + 1] = h[idx + 1]

# solve linear simultaneous equations.
Q, R = np.linalg.qr(A[:, :-1])
t = np.dot(Q.T, A[:, -1])
z_p = np.linalg.solve(R, t)

# fill intermidiate variable.
z = np.zeros(n, dtype=np.float32)
z[1:-1] = z_p

# calc polynomial coefficients.
coefficients = np.zeros((n - 1, dimention + 1), dtype=np.float32)
for i in xrange(n - 1):
    coefficients[i][0] = (z[i + 1] - z[i]) / (6 * h[i])
    coefficients[i][1] = (z[i] / 2)
    coefficients[i][2] = (b[i] -
                          h[i] *
                          (z[i + 1] + 2 * z[i]) / 6)
    coefficients[i][3] = sample_points_np[i][1]

# interpolate value.
axis = []
u = []
t = []

for i in xrange(n - 1):
    t = np.linspace(sample_points_np[i][0],
                    sample_points_np[i + 1][0], num=100)
    axis = np.append(axis, t)

    temp = t - sample_points_np[i][0]
    interpolated_value = 0

    for j in xrange(dimention + 1):
        interpolated_value += coefficients[i][j] * \
            np.power(temp, dimention - j)

    u = np.append(u, interpolated_value)

# visualize result
plt.plot(axis, u, 'b-')
plt.plot(sample_points_np[:, 0], sample_points_np[:, 1], 'ro')
plt.grid(True)

for i in range(0, n):
    point = sample_points_np[i].tolist()
    s = "({}, {})".format(*point)
    plt.annotate(s,
                 xy=tuple(point),
                 xytext=(point[0] + 0.25, point[1] + 1),)
plt.show()
