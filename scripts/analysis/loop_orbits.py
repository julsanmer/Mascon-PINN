import numpy as np
import os

from scripts.propagate_orbits import launch_propagation

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3

# Set configuration file
config = {
    'oe': {'a': np.linspace(28, 44, 9) * 1e3,
           'ecc': 0,
           'inc': np.array([0, 45, 90, 180]) * np.pi / 180,
           'omega': 48.2 * np.pi / 180,
           'RAAN': 347.8 * np.pi / 180,
           'f': 85.3 * np.pi / 180},
    't_prop': 24 * 3600,
    'groundtruth': {'file': '',
                    'asteroid_name': 'eros',  # 'eros'
                    'grav_model': 'poly',  # 'poly'
                    'file_poly': bsk_path + 'eros007790.tab',
                    'mascon': {'add': False,
                               'mu_M': np.array([0.1,
                                                 -0.1]) * 4.46275472004 * 1e5,
                               'xyz_M': np.array([[10, 0, 0],
                                                  [-10, 0, 0]]) * 1e3},
                    'dt_sample': 60},
    'estimation': {'file': 'mascon1000_muxyz_quadratic_octantrand0',
                   'model_path': '/ideal/dense_alt50km_100000samples/'}
}

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Model
model = 'mascon'
model = 'mascon-PINN'

# Number of masses
n_M = np.array([0,
                8,
                20,
                50,
                100,
                1000])
n_neurons = np.repeat(40, len(n_M))

# Propagate orbits
for i in range(len(n_M)):
    if model == 'mascon':
        # Set n_M
        config['estimation']['file'] = 'mascon' + str(n_M[i]) \
                                       + '_muxyz_quadratic_octantrand0'
    elif model == 'mascon-PINN':
        # Set neurons and n_M
        config['estimation']['file'] = 'pinn6x' + str(n_neurons[i]) \
                                        + 'SIREN_linear_mascon' + str(n_M[i])

    # Launch
    launch_propagation(config)
