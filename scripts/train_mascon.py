import numpy as np
import pickle as pck
import os

from src.scenario import Scenario
from plots.plots_gravity import all_gravityplots

# Import current directory
current_dir = os.getcwd()

# Conversion constants
km2m = 1e3


# This sets a configuration dict for mascon training
def configuration():
    # Set configuration file
    config = {
        'groundtruth': {'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': current_dir + '/Polyhedron_files/eros/'
                                     + 'eros007790.tab',
                        #'file_poly': bsk_path + 'eros007790.tab',
                        #'file_poly': bsk_path + 'ver128q.tab',
                        'n_face': [],  # to be filled later
                        'data': 'dense',  # 'dense' / 'orbit'
                        'mascon': {'add': False,
                                   'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'dense': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                                  'rmax': 50 * km2m,
                                  'n_data': 100000},
                        'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                                    'n_2D': 160, 'rmax_2D': 60 * km2m}},
        'regression': {'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                       'data': {'add_ejecta': False,  # bool
                                'n_ejecta': 50,
                                'dev_ejecta': 0,
                                'n_data': 100000},
                       'grad_descent': {'maxiter': 5000,
                                        'lr': 1e-3,
                                        'batch_size': 10000,
                                        'loss': 'quadratic',
                                        'train_xyz': True},
                       'mascon': {'train_xyz': True,  # bool
                                  'init': 'octant',
                                  'n_M': 100,
                                  'seed_M': 0,
                                  'file_shape': current_dir + '/Polyhedron_files/eros/'
                                                + 'EROS856Vert1708Fac.txt'}}
    }

    return config


# This loads data and train mascon based on config
def launch_training(config):
    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    gt = scenario.groundtruth
    gt.set_file(config['groundtruth'])
    gt.import_data(n_data=config['regression']['data']['n_data'])

    # Initialize regression instance
    scenario.init_regression()

    # Add mascon optimizer
    mascon_optim = scenario.grav_optimizer

    # Add shape and initialize distribution
    mascon_optim.add_poly_shape(config['regression']['mascon']['file_shape'])
    mascon_optim.init_distribution(config['regression']['mascon']['n_M'],
                                   gt.asteroid.mu)

    # Prepare mascon optimizer
    config_gd = config['regression']['grad_descent']
    mascon_optim.prepare_optimizer(loss=config_gd['loss'],
                                   maxiter=config_gd['maxiter'],
                                   lr=config_gd['lr'],
                                   batch_size=config_gd['batch_size'],
                                   xyzM_ad=np.array([16.3426, 8.41061, 5.973615])*km2m/10,
                                   muM_ad=gt.asteroid.mu / (config['regression']['mascon']['n_M']+1),
                                   train_xyz=config_gd['train_xyz'])

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
    asteroid = scenario.grav_optimizer.asteroid

    # Add regressed mascon
    mu_M = mascon_optim.mu_M
    xyz_M = mascon_optim.xyz_M
    asteroid.add_mascon(mu_M=mu_M, xyz_M=xyz_M)

    # Import gravity map grids
    gravmap = scenario.grav_optimizer.gravmap
    gravmap.import_grids(gt.gravmap)

    # Generate gravity maps and errors
    gravmap.generate_maps(asteroid)
    gravmap.error_maps(gt.gravmap)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(mascon_optim.file, "wb") as f:
        pck.dump(scenario, f)

    return scenario


if __name__ == "__main__":
    # Generate configuration
    config = configuration()

    # Launch training
    scenario = launch_training(config)

    # Plot gravity
    all_gravityplots(scenario)
