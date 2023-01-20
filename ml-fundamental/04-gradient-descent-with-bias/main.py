# Plot the reservations/pizzas dataset.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from function import train, predict

sns.set()

# loading the data and then calling the desired functions
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=10000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w, b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

