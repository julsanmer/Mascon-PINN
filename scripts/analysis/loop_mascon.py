import numpy as np
import os

from scripts.train_mascon import launch_training

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3


# Set configuration file
config = {
    'groundtruth': {'asteroid_name': 'eros',  # 'eros'
                    'grav_model': 'poly',
                    'file_poly': bsk_path + 'eros200700.tab',
                    #'file_poly': bsk_path + 'eros007790.tab',
                    #'file_poly': bsk_path + 'ver128q.tab',
                    'n_face': [],  # to be filled later
                    'data': 'dense',  # 'dense' / 'orbit'
                    'mascon': {'add': True,
                               'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                               'xyz_M': np.array([[10, 0, 0],
                                                  [-10, 0, 0]]) * 1e3},
                    'dense': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                              'rmax': 50 * km2m,
                              'n_data': 100000},
                    'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                                'n_2D': 160, 'rmax_2D': 60 * km2m}},
    'estimation': {'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                   'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                   'data': {'dmcukf_rate': 60,
                            'grav_rate': 60,
                            'orbits': np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                                [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]),
                            'add_ejecta': False,  # bool
                            'n_ejecta': 50,
                            'dev_ejecta': 0,
                            'n_data': 100000},
                   'grad_descent': {'maxiter': 5000,
                                    'lr': 1e-3,
                                    'batch_size': 2000,
                                    'loss': 'quadratic',
                                    'train_xyz': True},
                   'mascon': {'train_xyz': True,  # bool
                              'init': 'octant',
                              'n_M': 100,
                              'seed_M': 0}}
}

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Number of masses
n_M = np.array([0,
                8,
                20,
                50,
                100,
                1000])
#n_M = np.array([0])
#n_M = np.array([1])

# Train
for i in range(len(n_M)):
    # Set n_M
    config['estimation']['mascon']['n_M'] = n_M[i]

    # Launch
    launch_training(config)
