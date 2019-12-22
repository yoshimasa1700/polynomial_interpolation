import matplotlib.pyplot as plt
from lagrange_interpolation import lagrange_interpolation_function

# example data
# y = 2x + 1

data_points = [(1, 3), (2, 5)]
sample = [(x, lagrange_interpolation_function(data_points, x)) for x in xrange(0, 10)]

plt.scatter(*zip(*sample))
plt.show()
