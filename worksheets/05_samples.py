""" Problem set 5 - Samples """

# Problem 1: Create a twinned system by using Twin(hamiltonian, transform, second_component_weight)

# Supplementary problem: what transformation do you need to make the twin reflected about a given axis, e.g. (1,1,1) ? 
# Try importing the rotation matrix class? 

# Problem 2:
#   a) Create a powder spectrum of one of the systems we've looked at so far (simple might make things quicker)
#   b) Create a parameterisation of some exchanges, and make some simulated data for a specific value
#   c) Create an objective function for fitting
#   d) Do a "scan" of the objective function vs. parameter, try different n_points, how does it affect the optimisation landscape
#   e) Create a fit
#   f) Simulate the effect of noise
#   g) How does this effect the solution

""" Basic outline"""

from pyspinw import *
import numpy as np
from scipy.optimize import Bounds, least_squares # Can try other optimisers

#
# Define a system
#

unit_cell = UnitCell(3, 8, 8)

x = LatticeSite(0, 0, 0, 0, 1, 0, name="X")
y = LatticeSite(0.5, 0, 0, 0, 1, 0, name="Y")

j1 = HeisenbergExchange(x, y, j=1, cell_offset=(0, 0, 0), name="J1")
j2 = HeisenbergExchange(y, x, j=1, cell_offset=(0, 1, 0), name="J2")

sites = [x, y]
exchanges = [j1, j2]

s = Structure(sites, unit_cell)

hamiltonian = Hamiltonian(s, exchanges)

#
# Create powder sample
#

sample = Powder(hamiltonian)

path1D = Path1D(0.01, 1, n_points=20)

n_energy = 20
n_samples = 500

spectrum = sample.parameterized_spectrum(
    parameters=["J.j"],
    path=path1D,
    n_energy_bins=n_energy,
    n_samples=n_samples,
    energy_stddev=0.4,
    find_ground_state_with={"fixed": [x], "verbose": False})

target = spectrum(1.2)

#
# TODO: Synethesise noise here
#

def objective_lsq(x):
    return (target - spectrum(x)).reshape(-1)

def callback(intermediate_result):
    """ Callback to print out what is going on"""
    print(intermediate_result)

#
# Plot the fitting landscape
#

x = np.linspace(0.9, 1.3, 31)
import matplotlib.pyplot as plt
plt.plot(x, [objective_lsq(xx) for xx in x])
plt.show()

#
# Do an optimisation
#

print("Optimising...")

import time
t0 = time.time()

bounds = Bounds([0.5], [1.5])
solution = least_squares(objective_lsq, x0=1.0, bounds=bounds, callback=callback, diff_step=1e-2)

print(solution)
print(time.time() - t0)

