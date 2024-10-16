import numpy as np
import pickle as pck
import os

from src.groundtruth.groundtruth import Groundtruth
from src.gravRegression.mascon.masconOptimizer import MasconOptimizer
from plots.plots_regression import all_regression_plots

# Import current directory
current_dir = os.getcwd()

# Conversion constants
km2m = 1e3


# This sets a configuration dict for mascon training
def configuration():
    # Set configuration file
    config_gt = {'asteroid_name': 'eros',  # 'eros'
                 'grav_model': 'poly',
                 'file_poly': current_dir + '/Polyhedron_files/eros/'
                              + 'eros007790.tab',
                 #'file_poly': bsk_path + 'eros007790.tab',
                 #'file_poly': bsk_path + 'ver128q.tab',
                 'n_face': [],  # to be filled later
                 'data': 'dense',  # 'dense' / 'orbit'
                 'mascon': {'add': False,
                            'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                            'xyz_M': np.array([[8, 0, 0],
                                               [-8, 0, 0]]) * 1e3},
                 'dense': {'dist': 'rad',  # 'alt', 'rad', 'ell'
                           'rmax': 50 * km2m,
                           'n_data': 100000},
                 'gravmap': {'nr_3D': 60, 'nlat_3D': 60, 'nlon_3D': 60, 'rmax_3D': 160 * km2m,
                             'n_2D': 300, 'rmax_2D': 60 * km2m}}
    config_regression = {'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                         'data': {'add_ejecta': False,  # bool
                                  'n_ejecta': 50,
                                  'dev_ejecta': 0,
                                  'n_data': 50000},
                         'grad_descent': {'maxiter': 10,
                                          'lr': 1e-3,
                                          'batch_size': 10000,
                                          'loss': 'quadratic'},
                         'mascon': {'train_xyz': True,  # bool
                                    'init': 'octant',
                                    'n_M': 100,
                                    'seed_M': 0,
                                    'file_shape': current_dir + '/Polyhedron_files/eros/'
                                                  + 'EROS856Vert1708Fac.txt'}}

    return config_gt, config_regression


# This loads data and train mascon based on config
def launch_training(config_gt, config_regression):
    # Import groundtruth
    gt = Groundtruth()
    gt.set_file(config_gt)
    gt.import_data(n_data=config_regression['data']['n_data'])

    # Add mascon optimizer
    mascon_optim = MasconOptimizer()
    mascon_optim.initialize(config_gt, config_regression)

    # Add shape and initialize distribution
    mascon_optim.add_poly_shape(config_regression['mascon']['file_shape'])
    mascon_optim.init_distribution(config_regression['mascon']['n_M'],
                                   gt.asteroid.mu)

    # Prepare mascon optimizer
    config_gd = config_regression['grad_descent']
    mascon_optim.prepare_optimizer(loss=config_gd['loss'],
                                   maxiter=config_gd['maxiter'],
                                   lr=config_gd['lr'],
                                   batch_size=config_gd['batch_size'],
                                   xyzM_ad=np.array([16.3426, 8.41061, 5.973615])*km2m/10,
                                   muM_ad=gt.asteroid.mu / (config_regression['mascon']['n_M']+1),
                                   train_xyz=config_regression['mascon']['train_xyz'])

    # Retrieve data to train
    pos_data = gt.spacecraft.data.pos_BP_P
    acc_data = gt.spacecraft.data.acc_BP_P
    # pos_ejecta = groundtruth.ejecta.data.pos_BP_P
    # acc_ejecta = groundtruth.ejecta.data.acc_BP_P
    #
    # pos_data = np.concatenate((pos_data,
    #                            pos_ejecta))
    # acc_data = np.concatenate((acc_data,
    #                           acc_ejecta))

    # Train mascon distribution
    print('------- Initiating mascon fit -------')
    mascon_optim.optimize(pos_data, acc_data)
    mascon_optim.delete_optimizer()
    print('------- Finished mascon fit -------')

    # Save mascon model
    mascon_optim.save_model(path=mascon_optim.file)

    # Retrieve asteroid
    asteroid = mascon_optim.asteroid

    # Add regressed mascon
    mu_M = mascon_optim.mu_M
    xyz_M = mascon_optim.xyz_M
    asteroid.add_mascon(mu_M=mu_M, xyz_M=xyz_M)

    # Import gravity map grids
    gravmap = mascon_optim.gravmap
    gravmap.import_grids(gt.gravmap)

    # Generate gravity maps and errors
    gravmap.generate_maps(asteroid)
    gravmap.error_maps(gt.gravmap)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(mascon_optim.file, "wb") as f:
        pck.dump((gt, mascon_optim), f)

    return gt, mascon_optim


if __name__ == "__main__":
    # Generate configuration
    config_gt, config_regression = configuration()

    # Launch training
    gt, mascon_optim = launch_training(config_gt, config_regression)

    # Plot gravity
    all_regression_plots(gt, mascon_optim)
