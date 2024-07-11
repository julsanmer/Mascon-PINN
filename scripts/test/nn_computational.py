import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
from timeit import default_timer as timer

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

# Change working path
current_path = os.getcwd()
new_path = os.path.dirname(current_path)
os.chdir(new_path)

# get directory of this file
current_path = os.path.dirname(os.path.realpath(__file__)) 
THOR_PATH = os.path.abspath(current_path + '/../..')

import sys
sys.path.append(THOR_PATH)

# PINN and polyhedron paths
results_path = f'{THOR_PATH}/scripts/Results/eros/results/poly200700faces/ideal/dense_alt50km_100000samples/'
file_pinn = results_path + 'pinn6x40SIREN_linear_mascon1000.pck'
gt_path1 = f'{THOR_PATH}/scripts/Results/eros/groundtruth/poly200700faces/'
file_poly1 = gt_path1 + 'dense_alt50km_100000samples.pck'
gt_path2 = f'{THOR_PATH}/scripts/Results/eros/groundtruth/poly7790faces/'
file_poly2 = gt_path2 + 'dense_alt50km_100000samples.pck'

# Load groundtruth asteroids
inputs = pck.load(open(file_poly1, "rb"))
asteroid_poly1 = inputs.groundtruth.asteroid
asteroid_poly1.shape.create_shape()
asteroid_poly1.gravity[0].create_gravity()

inputs = pck.load(open(file_poly2, "rb"))
asteroid_poly2 = inputs.groundtruth.asteroid
asteroid_poly2.gravity[0].create_gravity()

# Load mascon-pinn asteroid
inputs = pck.load(open(file_pinn, "rb"))
asteroid_masconpinn = inputs.estimation.asteroid
asteroid_masconpinn.gravity[0].create_gravity()
asteroid_masconpinn.gravity[1].create_gravity()

# Create evaluation grid
n_samples = 1000
xyz_low = np.array([-40, -40, -40])*1e3
xyz_sup = np.array([40, 40, 40])*1e3
pos = np.zeros((n_samples, 3))
idx = 0
while idx < n_samples:
    pos_test = np.random.uniform(xyz_low, xyz_sup)
    is_exterior = asteroid_poly1.shape.check_exterior(pos_test)
    if is_exterior:
        pos[idx, 0:3] = pos_test
        idx += 1

# Preallocate cpu times
t_bsk = np.zeros(n_samples) #
t_bsk2 = np.zeros(n_samples) #
t_grad = np.zeros(n_samples) # PINN in python layer
t_mascon = np.zeros(n_samples) # Mascon
t_poly1 = np.zeros(n_samples) # Polyhedron high
t_poly2 = np.zeros(n_samples) # Polyhedron low

# Loop evaluation samples
for i in range(n_samples):
    # Polyhedron evaluation (200700 faces)
    t_start = timer()
    acc = asteroid_poly1.compute_gravity(pos[i, 0:3])
    t_end = timer()
    t_poly1[i] = t_end - t_start

    # Polyhedron evaluation (7790 faces)
    t_start = timer()
    acc = asteroid_poly2.compute_gravity(pos[i, 0:3])
    t_end = timer()
    t_poly2[i] = t_end - t_start

    # Only PINN evaluation (from BSK)
    t_start = timer()
    acc = asteroid_masconpinn.gravity[0].compute_acc(pos[i, 0:3])
    t_end = timer()
    t_bsk[i] = t_end - t_start

    # Only mascon evaluation (from BSK)
    t_start = timer()
    acc = asteroid_masconpinn.gravity[1].compute_acc(pos[i, 0:3])
    t_end = timer()
    t_mascon[i] = t_end - t_start

    # Mascon+PINN evaluation (from BSK)
    t_start = timer()
    acc = asteroid_masconpinn.compute_gravity(pos[i, 0:3])
    t_end = timer()
    t_bsk2[i] = t_end - t_start

    # PINN evaluation (from Python)
    t_start = timer()
    acc = asteroid_masconpinn.gravity[0].network_eval.gradient(np.array([pos[i, 0:3]]))
    t_end = timer()
    t_grad[i] = t_end - t_start

# Do plots in miliseconds
plt.plot(t_poly1*1e3, marker='.', label='Poly. 200700 faces')
plt.plot(t_poly2*1e3, marker='.', label='Poly. 7790 faces')
plt.plot(t_bsk*1e3, marker='.', label='PINN BSK')
plt.plot(t_bsk2*1e3, marker='.', label='Mascon+PINN BSK')
plt.plot(t_grad*1e3, marker='.', label='PINN Python')
plt.plot(t_mascon*1e3, marker='.', label='Mascon')
plt.semilogy()
plt.legend()
plt.show()
