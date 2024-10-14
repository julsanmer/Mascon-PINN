import numpy as np
import pickle as pck
import os

from plots.plots_regression import all_regression_plots

# Import current directory
current_dir = os.path.dirname(os.getcwd())

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
                 'mascon': {'add': True,
                            'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                            'xyz_M': np.array([[8, 0, 0],
                                               [-8, 0, 0]]) * 1e3},
                 'dense': {'dist': 'rad',  # 'alt', 'rad', 'ell'
                           'rmax': 50 * km2m,
                           'n_data': 100000},
                 'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                             'n_2D': 160, 'rmax_2D': 60 * km2m}}
    config_regression = {'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                         'data': {'add_ejecta': False,  # bool
                                  'n_ejecta': 50,
                                  'dev_ejecta': 0,
                                  'n_data': 50000},
                         'grad_descent': {'maxiter': 5000,
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


if __name__ == "__main__":
    # Generate configuration
    config_gt, config_regression = configuration()

    # Load regressed mascon
    file_mascon = current_dir \
                  + '/Results/eros/regression/polyheterogeneous7790faces/ideal' \
                  + '/dense_rad50km_50000samples/mascon100_muxyz_quadratic_octantrand0.pck'
    gt, mascon_optim = pck.load(open(file_mascon, "rb"))

    # Plot gravity
    all_regression_plots(gt, mascon_optim)
