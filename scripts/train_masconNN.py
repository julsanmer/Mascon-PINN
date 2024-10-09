import numpy as np
import pickle as pck
import os

from src.groundtruth.groundtruth import Groundtruth
from src.gravRegression.nn.nnOptimizer import NNOptimizer
from plots.plots_gravity import all_gravityplots

# Import current directory
current_dir = os.getcwd()

# Conversion constants
km2m = 1e3


# This sets a configuration dict for mascon-PINN training
def configuration():
    # Set configuration file
    config_gt = {'file': '',  # filled automatically
                 'asteroid_name': 'eros',  # 'eros'
                 'grav_model': 'poly',
                 'file_poly': current_dir + '/Polyhedron_files/eros/'
                              + 'eros007790.tab',
                 # 'file_poly': bsk_path + 'eros007790.tab',
                 # 'file_poly': bsk_path + 'ver128q.tab',
                 'n_face': [],  # to be filled later
                 'data': 'dense',  # 'dense' / 'orbit'
                 'mascon': {'add': True,
                            'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                            'xyz_M': np.array([[10, 0, 0],
                                               [-10, 0, 0]]) * 1e3},
                 'dense': {'dist': 'rad',  # 'alt', 'rad', 'ell'
                           'rmax': 50 * km2m,
                           'n_data': 100000},
                 'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                             'n_2D': 160, 'rmax_2D': 60 * km2m}}
    config_regression = {'file': '',
                         'grav_model': 'pinn',  # 'mascon' / 'pinn' / 'spherharm'
                         'data': {'n_data': 50000},
                         'grad_descent': {'maxiter': 5000,
                                          'lr': 4*1e-3,
                                          'batch_size': 1000,
                                          'loss': 'linear'},
                         'nn': {'model': 'NN',  # 'PINN' / 'NN'
                                'neurons': 40,
                                'layers': 6,
                                'activation': 'GELU',  # 'GELU' / 'SiLU' / 'SIREN'
                                'R': 16*2 * km2m,
                                'l_bc': 1,
                                 #'eval_mode': 'BSK',  # 'BSK' / 'autograd'
                                'model_bc': {'file': 'mascon100_muxyz_quadratic_octantrand0.pck',
                                             'model': 'mascon',
                                             'n_M': 100}}}

    return config_gt, config_regression


# This loads data and train mascon based on config
def launch_training(config_gt, config_regression):
    # Get gradient descent and pinn
    config_gd = config_regression['grad_descent']
    config_nn = config_regression['nn']

    # Import groundtruth
    gt = Groundtruth()
    gt.set_file(config_gt)
    gt.import_data(n_data=config_regression['data']['n_data'])

    # Initialize estimation
    nn_optim = NNOptimizer()
    nn_optim.initialize(config_gt, config_regression)

    # Import mascon model
    file_mascon = '/'.join(nn_optim.file.split('/')[:-1]) \
                  + '/' + config_nn['model_bc']['file']
    _, input_mascon = pck.load(open(file_mascon, "rb"))
    mu_M = input_mascon.asteroid.gravity[0].mu_M
    xyz_M = input_mascon.asteroid.gravity[0].xyz_M

    # Prepare pinn optimizer
    nn_optim.init_network(n_layers=config_nn['layers'],
                          n_neurons=config_nn['neurons'],
                          activation=config_nn['activation'],
                          model=config_nn['model'])
    nn_optim.set_extra_params(R=config_nn['R'],
                              l_bc=config_nn['l_bc'])
    nn_optim.prepare_optimizer(maxiter=config_gd['maxiter'],
                               lr=config_gd['lr'],
                               batch_size=config_gd['batch_size'],
                               loss_type=config_gd['loss'])

    # Copy mascon
    nn_optim.add_mascon(mu_M, xyz_M)

    # Retrieve data to train
    pos_data = gt.spacecraft.data.pos_BP_P
    acc_data = gt.spacecraft.data.acc_BP_P
    U_data = gt.spacecraft.data.U
    # pos_ejecta = groundtruth.ejecta.data.pos_BP_P
    # acc_ejecta = groundtruth.ejecta.data.acc_BP_P
    # U_ejecta = groundtruth.ejecta.data.U
    #
    # pos_data = np.concatenate((pos_data,
    #                            pos_ejecta))
    # acc_data = np.concatenate((acc_data,
    #                           acc_ejecta))
    # U_data = np.concatenate((U_data,
    #                         U_ejecta))

    # Train nn
    print('------- Initiating pinn training -------')
    nn_optim.optimize(pos_data, acc_data, U_data)
    print('------- Finished pinn training -------')

    # Save pinn model
    nn_optim.save_model(path=nn_optim.file)

    # Retrieve asteroid
    asteroid = nn_optim.asteroid

    # Add trained pinn
    asteroid.add_nn(file_torch=nn_optim.file_torch)

    # # Add regressed mascon
    # mu_M = pinn_optim.grav_nn.model_bc.mu_M.numpy().astype(np.float64)
    # xyz_M = pinn_optim.grav_nn.model_bc.xyz_M.numpy().astype(np.float64)
    # asteroid.add_mascon(mu_M=mu_M, xyz_M=xyz_M)

    # Import gravity map grids
    gravmap = nn_optim.gravmap
    gravmap.import_grids(gt.gravmap)

    # Generate gravity maps and errors
    gravmap.generate_maps(asteroid)
    gravmap.error_maps(gt.gravmap)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(nn_optim.file, "wb") as f:
        pck.dump((gt, nn_optim), f)

    return gt, nn_optim


if __name__ == "__main__":
    # Generate configuration
    config_gt, config_nn = configuration()

    # Launch training
    gt, nn_optim = launch_training(config_gt, config_nn)

    # Plot gravity
    all_gravityplots(gt, nn_optim)
