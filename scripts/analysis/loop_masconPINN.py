import numpy as np
import os

from scripts.train_masconPINN import launch_training

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3

# Set configuration file
config = {
    'groundtruth': {'file': '',  # filled automatically
                    'asteroid_name': 'eros',  # 'eros'
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
    'estimation': {'file': '',
                   'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                   'grav_model': 'pinn',  # 'mascon' / 'pinn' / 'spherharm'
                   'data': {'n_data': 100000},
                   'grad_descent': {'maxiter': 5000,
                                    'lr': 1e-3,
                                    'batch_size': 2000,
                                    'loss': 'linear'},
                   'pinn': {'n_inputs': 5,
                            'neurons': 40,
                            'layers': 6,
                            'activation': 'SIREN',  # 'GELU' / 'SiLU' / 'SIREN'
                            'r_ad': 3.47 * km2m,
                            'switch': {'rad_bc': 16. * km2m,
                                       'r_bc': 70. * km2m,
                                       'k_bc': 1.,
                                       'l_bc': 1},
                            'eval_mode': 'BSK',  # 'BSK' / 'autograd'
                            'model_bc': {'file': 'mascon100_muxyz_quadratic_octantrand0.pck',
                                         'type': 'mascon',
                                         'n_M': 100}}}
}



# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Number of masses and neurons
# n_M = np.array([0,
#                 8,
#                 20,
#                 50,
#                 100,
#                 1000])
# n_neurons = np.repeat(40, len(n_M))

# # Number of masses and neurons
n_neurons = np.array([10,
                      20,
                      80,
                      160])
n_M = np.repeat(100, len(n_neurons))

# Train
for i in range(len(n_M)):
    # Set n_M
    config['estimation']['pinn']['model_bc']['file'] = 'mascon' + str(n_M[i]) + '_muxyz_quadratic_octantrand0.pck'
    config['estimation']['pinn']['model_bc']['n_M'] = n_M[i]

    # Set neurons
    config['estimation']['pinn']['neurons'] = n_neurons[i]

    # Launch
    launch_training(config)
