# Plot the reservations/pizzas dataset.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from function import train, predict

sns.set()

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)

# plt.plot(X, Y, "bo")
# plt.xlabel("Reservations")
# plt.ylabel("Pizzas")
# x_edge, y_edge = 50, 50
# plt.axis([0, x_edge, 0, y_edge])
# plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
# plt.show()

