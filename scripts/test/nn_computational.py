import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os

from timeit import default_timer as timer

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

file_path = 'Results/eros/results/poly/ideal/dense_alt50km_100000samples/'
file_pinn = file_path + 'pinn6x40SIREN_linear_mascon1000.pck'
filepoly_path = 'Results/eros/groundtruth/poly/'
file_poly = filepoly_path + 'dense_alt50km_100000samples.pck'

inputs = pck.load(open(file_poly, "rb"))
asteroid_poly = inputs.groundtruth.asteroid
asteroid_poly.gravity[0].file = bsk_path + 'ver128q.tab'
asteroid_poly.gravity[0].create_gravity()

inputs = pck.load(open(file_pinn, "rb"))
asteroid = inputs.estimation.asteroid
asteroid.shape.create_shape()

asteroid.gravity[0].create_gravity()
asteroid.gravity[1].create_gravity()

n_samples = 1000

xyz_low = np.array([-40, -40, -40])*1e3
xyz_sup = np.array([40, 40, 40])*1e3
pos = np.zeros((n_samples, 3))
idx = 0
while idx < n_samples:
    pos_test = np.random.uniform(xyz_low, xyz_sup)
    is_exterior = asteroid.shape.check_exterior(pos_test)

    if is_exterior:
        pos[idx, 0:3] = pos_test
        idx += 1

t_bsk = np.zeros(n_samples)
t_bsk2 = np.zeros(n_samples)
t_grad = np.zeros(n_samples)
t_mascon = np.zeros(n_samples)
t_poly = np.zeros(n_samples)

for i in range(n_samples):
    # Start measuring cpu time
    t_start = timer()

    # BSK acceleration
    acc = asteroid_poly.compute_gravity(pos[i, 0:3])

    # End measuring cpu time
    t_end = timer()
    t_poly[i] = t_end - t_start

for i in range(n_samples):
    # Start measuring cpu time
    t_start = timer()

    # BSK acceleration
    acc = asteroid.gravity[0].compute_acc(pos[i, 0:3])

    # End measuring cpu time
    t_end = timer()
    t_bsk[i] = t_end - t_start

for i in range(n_samples):
    # Start measuring cpu time
    t_start = timer()

    # BSK acceleration
    acc = asteroid.gravity[1].compute_acc(pos[i, 0:3])

    # End measuring cpu time
    t_end = timer()
    t_mascon[i] = t_end - t_start

for i in range(n_samples):
    # Start measuring cpu time
    t_start = timer()

    # BSK acceleration
    acc = asteroid.compute_gravity(pos[i, 0:3])

    # End measuring cpu time
    t_end = timer()
    t_bsk2[i] = t_end - t_start

for i in range(n_samples):
    # Start measuring cpu time
    t_start = timer()

    # Autograd acceleration
    acc = asteroid.gravity[0].network_eval.gradient(np.array([pos[i, 0:3]]))

    # End measuring cpu time
    t_end = timer()
    t_grad[i] = t_end - t_start

plt.plot(t_poly, marker='.')
plt.plot(t_bsk, marker='.')
plt.plot(t_bsk2, marker='.')
plt.plot(t_grad, marker='.')
plt.plot(t_mascon, marker='.')
plt.semilogy()
plt.show()
