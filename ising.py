import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.ndimage import convolve, generate_binary_structure
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.constants as constants

""" Ising Model Function List (tbc) """

# =============================================================================
# # Metropolis Algorithm
# =============================================================================

def metropolis(arr=lattice, beta=beta):

    """
    >>Metropolis-Hastings Algorithm for a 2D Ising Model<<

    The objective is to find an equilibrium state at a particular temperature
    where by which we start with a random lattice of spins and use
    nearest neighbour calculation to flip the spin of a randomly chosen point.

    1) Start with an initial NxN lattice/configuration of spins

    2) Flip the spin of a randomly chosen point on the lattice

    3) We want to find the probability that we will accept this new state
    using configuration probability

    4) Calculate the change in energy, dE

    5) if dE < 0 or a randomly chosen small value is less than the
    configuration probability, accept the flip.

    6) Continue over chosen number of iterations to reach equilibrium

    """

    for i in range(iterations):
        for j in range(iterations):
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)

            copy = arr[x, y]
            nn = (
                arr[(x + 1) % N, y]
                + arr[x, (y + 1) % N]
                + arr[(x - 1) % N, y]
                + arr[x, (y - 1) % N]
            )

            nn = 2 * copy * nn
            if nn < 0 or rand() < np.exp(-nn * beta):
                copy *= -1

            arr[x, y] = copy
    return arr

def animate(image):
    # Clear axis
    plt.cla()
    # Plot using Metropolis Algorithm
    plot = plt.pcolormesh(metropolis(), cmap="plasma") #viridis
    
    return plot
