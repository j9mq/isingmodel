import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.ndimage import convolve, generate_binary_structure
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# # Parameters
# =============================================================================
N = 100  # NxN lattice
beta = 10  # 10 most optimised value
iterations = int(1024)  # 5e3 most optimised value

kB = 1.380649e-23  # Boltzmann constant
nt = 88  # Number of temperature points
T = np.linspace(1.53, 3.28, nt)
# Temperature
E, M, C, X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2 = 1.0 / (iterations * np.power(N, 2)), 1.0 / (
    np.power(iterations, 2) * np.power(N, 2)
)


# Initialize a random configuration for spins (1 or -1)
lattice = np.random.choice([-1, 1], size=(N, N))

# =============================================================================
# # Calculations
# =============================================================================


def energy_arr(lattice=lattice):
    """Finds energy array and
    applies the nearest neighbours summation"""
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode="constant", cval=0)
    return arr


def energy_calc(lattice=lattice):
    """Sums energy array"""
    return energy_arr(lattice).sum() * -1 / 2


def mag_calc(lattice=lattice):
    """Magnetization of a given configuration"""
    mag = np.sum(lattice)
    return mag


def c_calc(lattice=lattice):
    """Specific Heat of a given configuration"""
    c = np.var(energy_arr())
    return c


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


# =============================================================================
# # Visualization
# =============================================================================

plt.rcParams["figure.dpi"] = 400
plt.rcParams.update({"font.size": 9})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# plt.imshow(lattice, cmap="binary", vmin=-1, vmax=1)
plt.pcolormesh(lattice, cmap="viridis")
plt.title("Initial lattice configuration")

plt.show()

# plt.imshow(metropolis(), cmap="binary", vmin=-1, vmax=1)
# plt.pcolormesh(metropolis(), cmap="viridis")
# plt.title("Final lattice configuration")

plt.show()

# =============================================================================
# # Animation
# =============================================================================

"""

def animate(image):
    # Clear axis
    plt.cla()
    # Plot using Metropolis Algorithm
    plot = plt.pcolormesh(metropolis(), cmap="plasma") #viridis
    
    return plot

fig = plt.figure(figsize = (6,6))    

anim = FuncAnimation(fig, func=animate, frames=150)
anim.save("metropolis_500x500_1000i.gif", writer=PillowWriter(fps=10))

"""

# =============================================================================
# # Calculation II
# =============================================================================

for tt in range(nt):
    E1 = M1 = E2 = M2 = 0
    iT = 1.0 / T[tt]
    iT2 = iT * iT

    for i in range(iterations):  # equilibrate
        metropolis(lattice, iT)  # Monte Carlo moves

    for i in range(iterations):
        metropolis(lattice, iT)
        Ene = energy_calc(lattice)  # calculate the energy
        Mag = mag_calc(lattice)  # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag * Mag
        E2 = E2 + Ene * Ene

    E[tt] = n1 * E1
    M[tt] = n1 * M1
    C[tt] = (n1 * E2 - n2 * E1 * E1) * iT2
    X[tt] = (n1 * M2 - n2 * M1 * M1) * iT

# =============================================================================
# # Visualisation II (Plotting)
# =============================================================================

f = plt.figure(figsize=(18, 10))
# plot the calculated values

sp = f.add_subplot(2, 2, 1)
plt.scatter(T, E, s=50, marker="o", color="lightseagreen")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Energy ", fontsize=20)
plt.axis("tight")

sp = f.add_subplot(2, 2, 2)
plt.scatter(T, abs(M), s=50, marker="o", color="darkslateblue")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetization ", fontsize=20)
plt.axis("tight")

sp = f.add_subplot(2, 2, 3)
plt.scatter(T, C, s=50, marker="o", color="lightseagreen")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Specific Heat ", fontsize=20)
plt.axis("tight")

sp = f.add_subplot(2, 2, 4)
plt.scatter(T, X, s=50, marker="o", color="darkslateblue")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Susceptibility", fontsize=20)
plt.axis("tight")
