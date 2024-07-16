import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
import torch
from timeit import default_timer as timer

def main():
    # get directory of this file
    current_path = os.path.dirname(os.path.realpath(__file__)) 
    THOR_PATH = os.path.abspath(current_path + '/../..')

    import sys
    sys.path.append(THOR_PATH)

    # PINN and polyhedron paths
    results_path = f'{THOR_PATH}/scripts/Results/eros/results/poly200700faces/ideal/dense_alt50km_100000samples/'
    file_pinn, size = results_path + 'pinn6x40SIREN_linear_mascon1000.pck', 40
    # file_pinn, size = results_path + 'pinn6x40SIREN_linear_mascon100.pck', 40
    # file_pinn, size = results_path + 'pinn6x20SIREN_linear_mascon100.pck', 20

    # Load mascon-pinn asteroid
    inputs = pck.load(open(file_pinn, "rb"))
    asteroid_masconpinn = inputs.estimation.asteroid
    asteroid_masconpinn.gravity[0].create_gravity()
    asteroid_masconpinn.gravity[1].create_gravity()

    # Create evaluation grid
    n_samples = 1000
    pos = np.random.uniform(-40_000, 40_000, size=(n_samples, 3))

    # Preallocate cpu times
    t_bsk = np.zeros(n_samples) #
    t_bsk2 = np.zeros(n_samples) #
    t_grad = np.zeros(n_samples) # PINN in python layer
    t_JIT = np.zeros(n_samples) # PINN in python layer
    t_JIT_II = np.zeros(n_samples) # PINN in python layer
    t_JIT_III = np.zeros(n_samples) # PINN in python layer
    t_JIT_IV = np.zeros(n_samples) # PINN in python layer
    t_JIT_V = np.zeros(n_samples) # PINN in python layer
    t_JIT_VI = np.zeros(n_samples) # PINN in python layer
    t_JIT_VII = np.zeros(n_samples) # PINN in python layer
    t_JIT_no_gradient = np.zeros(n_samples) # PINN in python layer
    t_mascon = np.zeros(n_samples) # Mascon

    # generate a JIT compiled PINN
    PINN_model = asteroid_masconpinn.gravity[0]

    # WARNING: There is not a trivial way to re-run init from the trained network
    pinn = PINN_model.network_eval
    pinn.ones = torch.ones([1, 1]).to("cpu")
    pinn_JIT = torch.compile(pinn)

    # Loop evaluation samples
    for i in range(n_samples):

        # Only PINN evaluation (from BSK)
        t_start = timer()
        PINN_model.compute_acc(pos[i, 0:3])
        t_end = timer()
        t_bsk[i] = t_end - t_start

        # Only mascon evaluation (from BSK)
        t_start = timer()
        asteroid_masconpinn.gravity[1].compute_acc(pos[i, 0:3])
        t_end = timer()
        t_mascon[i] = t_end - t_start

        # Mascon+PINN evaluation (from BSK)
        t_start = timer()
        asteroid_masconpinn.compute_gravity(pos[i, 0:3])
        t_end = timer()
        t_bsk2[i] = t_end - t_start

        # PINN evaluation (from Python)
        t_start = timer()
        PINN_model.network_eval.gradient(np.array([pos[i, 0:3]]))
        t_end = timer()
        t_grad[i] = t_end - t_start
        
        # PINN JIT (from Python)
        t_start = timer()
        pinn_JIT.gradient(np.array([pos[i, 0:3]]))
        t_end = timer()
        t_JIT[i] = t_end - t_start

        tensor = torch.tensor([pos[i, 0:3]])
        t_start = timer()
        pinn_JIT.gradient_II(tensor)
        t_end = timer()
        t_JIT_II[i] = t_end - t_start
        
        t_start = timer()
        pinn_JIT.gradient_III(tensor)
        t_end = timer()
        t_JIT_III[i] = t_end - t_start

        t_start = timer()
        pinn_JIT.gradient_IV(tensor)
        t_end = timer()
        t_JIT_IV[i] = t_end - t_start
        
        t_start = timer()
        pinn_JIT.gradient_V(tensor)
        t_end = timer()
        t_JIT_V[i] = t_end - t_start

        t_start = timer()
        pinn_JIT.gradient_VI(tensor)
        t_end = timer()
        t_JIT_VI[i] = t_end - t_start

        tensor = torch.tensor([pos[i, 0:3]], dtype=torch.float32)
        t_start = timer()
        pinn_JIT.gradient_VII(tensor)
        t_end = timer()
        t_JIT_VII[i] = t_end - t_start

        torch.tensor(pos[i, 0:3])
        t_start = timer()
        pinn_JIT.no_gradient(tensor)
        t_end = timer()
        t_JIT_no_gradient[i] = t_end - t_start

    # Do plots in miliseconds
    # plt.plot(t_bsk*1e3, marker='.', label='PINN BSK')
    # plt.plot(t_bsk2*1e3, marker='.', label='Mascon+PINN BSK')
    # plt.plot(t_mascon*1e3, marker='.', label='Mascon')
    plt.figure(figsize=(6, 3))
    plt.plot(t_grad*1e3, marker='.', label='PINN Python')
    plt.plot(t_JIT*1e3, marker='.', linestyle = '--', label='PINN JIT I')
    plt.plot(t_JIT_II*1e3, marker='.', linestyle = '--', label='PINN JIT II')
    plt.plot(t_JIT_III*1e3, marker='.', linestyle = '--', label='PINN JIT III')
    plt.plot(t_JIT_IV*1e3, marker='.', linestyle = '--', label='PINN JIT IV')
    plt.plot(t_JIT_V*1e3, marker='.', linestyle = '--', label='PINN JIT V')
    plt.plot(t_JIT_VI*1e3, marker='.', linestyle = '--', label='PINN JIT VI')
    plt.plot(t_JIT_VII*1e3, marker='.', linestyle = '--', label='PINN JIT VII')
    plt.plot(t_JIT_no_gradient*1e3, marker='.', linestyle = '--', label='PINN JIT No Gradient')
    plt.semilogy()
    plt.ylim(0.05, 1)
    plt.legend()
    plt.grid(True, which='both', axis='both')

    # save the file to the scripts/Plots directory
    plt.savefig(f'{current_path}/../Plots/nn_inference_test_{size}.png')

    a_JIT_I = pinn_JIT.gradient(tensor)
    a_JIT_II = pinn_JIT.gradient_II(tensor)
    a_JIT_III = pinn_JIT.gradient_III(tensor)


    assert np.allclose(a_JIT_I, a_JIT_II)
    assert np.allclose(a_JIT_I, a_JIT_III)

    plt.show()

if __name__ == '__main__':
    main()