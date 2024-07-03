import numpy as np
import pickle as pck

from src.orbits.orbits import Orbits
from plots.plots_orbits import all_orbitplots

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3


# This sets a configuration dict for orbits propagation
def configuration():
    # Set configuration file
    config = {
        'oe': {#'a': np.linspace(28, 46, 4) * 1e3,
               'a': np.array([34]) * km2m,
               'ecc': 0,
               'inc': np.array([0]),
               #'inc': np.array([0, 45, 90, 135, 180]) * np.pi/180,
               'omega': 48.2 * np.pi/180,
               'RAAN': 347.8 * np.pi/180,
               'f': 85.3 * np.pi/180},
        't_prop': 24 * 3600,
        'groundtruth': {'file': '',
                        'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',  # 'poly'
                        #'file_poly': bsk_path + 'eros007790.tab',
                        'file_poly': bsk_path + 'eros200700.tab',
                        'n_faces': [],
                        'mascon': {'add': False,
                                   'mu_M': np.array([0.1,
                                                     -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'dt_sample': 60},
        'estimation': {'file': '',
                        #'file': 'mascon1000_muxyz_quadratic_octantrand0',
                       'model_path': '/ideal/dense_alt50km_100000samples/'}
    }

    return config


# This function propagates orbits
def launch_propagation(config):
    # Create orbits class
    orbits = Orbits(config)
    config_gt = config['groundtruth']

    # Get scenario and model files
    file_ref = orbits.set_filegroundtruth(config_gt)

    # If estimation file is empty, do groundtruth simulation
    if config['estimation']['file'] == '':
        # Get file
        file_save = file_ref

        # Create asteroid
        asteroid = orbits.asteroid
        orbits.asteroid.set_properties(config_gt['asteroid_name'],
                                       file_shape=config_gt['file_poly'])
        orbits.asteroid.add_poly(config_gt['file_poly'])
        if config_gt['mascon']['add']:
            asteroid.add_mascon(config_gt['mascon']['mu_M'],
                                config_gt['mascon']['xyz_M'])
    else:
        file_save, file_model = \
            orbits.set_fileresults(config)

        # Retrieve asteroid
        inputs = pck.load(open(file_model, "rb"))
        asteroid = inputs.estimation.asteroid
        asteroid.shape.create_shape()
        orbits.asteroid = asteroid

    # Propagate and compute errors
    orbits.propagate()
    if config['estimation']['file'] != '':
        orbits.compute_errors(file_ref)

    # Save simulation outputs
    with open(file_save, "wb") as f:
        pck.dump(orbits, f)

    return orbits


if __name__ == "__main__":
    # Get configuration file
    config = configuration()

    # Get scenario and model files
    orbits = launch_propagation(config)

    # Plot
    all_orbitplots(orbits, config)
