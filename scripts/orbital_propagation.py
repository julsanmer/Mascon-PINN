import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os

from Basilisk.simulation import gravityEffector

from src.orbiters.spacecraft import Spacecraft
from plots.plots_orbits import plot_orb

# Import current directory
current_dir = os.getcwd()

# Conversion constants
km2m = 1e3


# This function defines a configuration dict
def configuration():
    # Set configuration file
    config = {
        'oe': {'a': np.linspace(28, 46, 4) * 1e3,
               'ecc': 0,
               'inc': np.array([0, 45, 90, 135, 180]) * np.pi/180,
               'omega': 48.2 * np.pi / 180,
               'RAAN': 347.8 * np.pi / 180,
               'f': 85.3 * np.pi / 180},
        'tf': 24 * 3600,
        'groundtruth': {'file': '',
                        'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',  # 'poly'
                        'file_poly': current_dir + '/Polyhedron_files/eros/'
                                     + 'eros007790.tab',
                        'mascon': {'add': False,
                                   'mu_M': np.array([0.1,
                                                     -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'dt_sample': 60},
        'regression': {'file': 'mascon100_muxyz_quadratic_octantrand0',
                       'model_path': '/ideal/dense_alt50km_100000samples/'}
    }

    return config


# Sets groundtruth file
def set_filegroundtruth(config_gt):
    # Set asteroid name and groundtruth gravity
    asteroid_name = config_gt['asteroid_name']

    # Create asteroid folder if it does not exist
    path_asteroid = 'Results/' + asteroid_name
    exist = os.path.exists(path_asteroid)
    if not exist:
        os.makedirs(path_asteroid)

    # Retrieve groundtruth parameters
    grav_gt = config_gt['grav_model']
    if config_gt['mascon']['add']:
        grav_gt += 'heterogeneous'

    # Obtain number of faces
    _, _, _, n_face = \
        gravityEffector.loadPolyFromFileToList(config_gt['file_poly'])
    config_gt['n_face'] = n_face

    # Create ground truth path if it does not exist and define ground truth file
    path = path_asteroid + '/groundtruth/' + grav_gt + str(n_face) + 'faces'
    exist = os.path.exists(path)
    if not exist:
        os.makedirs(path)
    file = path + '/propagation.pck'

    return file


# Sets results file
def set_fileresults(config):
    # Retrieve groundtruth config
    config_gt = config['groundtruth']

    # Set asteroid name and groundtruth gravity
    asteroid_name = config_gt['asteroid_name']

    # Create asteroid folder if it does not exist
    path_asteroid = current_dir + '/Results/' + asteroid_name
    exist = os.path.exists(path_asteroid)
    if not exist:
        os.makedirs(path_asteroid)

    # Retrieve groundtruth parameters
    grav_gt = config_gt['grav_model']
    if config_gt['mascon']['add']:
        grav_gt += 'heterogeneous'

    # Obtain number of faces
    _, _, _, n_face = \
        gravityEffector.loadPolyFromFileToList(config_gt['file_poly'])
    config_gt['n_face'] = n_face

    # Set results file
    path = path_asteroid + '/results/' + grav_gt + str(n_face) + 'faces'
    file = path + config['regression']['model_path'] \
           + config['regression']['file'] + '_orbits.pck'
    file_model = path + config['regression']['model_path'] \
                 + config['regression']['file'] + '.pck'

    return file, file_model


# This function launches orbital propagation
def launch_propagation(config):
    # Number of semi-major axes and
    # orbital inclinations
    n_a = len(config['oe']['a'])
    n_inc = len(config['oe']['inc'])

    # Retrieve propagation time
    tf = config['tf']

    # Files
    file, file_model = set_fileresults(config)

    # Create asteroid from a previously regressed model
    inputs = pck.load(open(file_model, "rb"))
    asteroid = inputs.grav_optimizer.asteroid
    asteroid.shape.create_shape()

    # Create a 2D list to store spacecraft objects
    sc_orbits = [[Spacecraft(grav_body=asteroid) for _ in range(n_inc)] for _ in range(n_a)]

    # Prepare initial orbital element
    oe0 = np.array([np.nan,
                    config['oe']['ecc'],
                    np.nan,
                    config['oe']['omega'],
                    config['oe']['RAAN'],
                    config['oe']['f']])

    # Propagation loop
    for i in range(n_a):
        # Change semi-major axis
        oe0[0] = config['oe']['a'][i]

        for j in range(n_inc):
            # Change inclination
            oe0[2] = config['oe']['inc'][j]

            # Propagate
            sc_orbits[i][j].propagate(oe0, tf)

    # Delete swigpy
    for i in range(n_a):
        for j in range(n_inc):
            # Delete swigpy
            sc_orbits[i][j].grav_body.delete_swigpy()

    # Save file
    with open(file, "wb") as f:
        pck.dump(sc_orbits, f)

    return sc_orbits


if __name__ == "__main__":
    # Obtain configuration dict
    config = configuration()

    # Launch orbital propagation
    sc_orbits = launch_propagation(config)

    # Plot orbits
    plot_orb(sc_orbits)
    plt.show()
