import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os

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
                        'mascon': {'add': True,
                                   'mu_M': np.array([0.1,
                                                     -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'dt_sample': 60},
        'regression': {'file': 'mascon100_muxyz_quadratic_octantrand0.pck',
                       'model_path': 'poly7790faces/ideal/dense_alt50km_100000samples/'}
    }

    return config


if __name__ == "__main__":
    # Obtain configuration dict
    config = configuration()

    # Number of semi-major axes and
    # orbital inclinations
    n_a = len(config['oe']['a'])
    n_inc = len(config['oe']['inc'])

    # Retrieve propagation time
    tf = config['tf']

    # Create asteroid from a previously regressed model
    scenario_mascon = pck.load(open('Results/eros/results/'
                                    + config['regression']['model_path']
                                    + config['regression']['file'], "rb"))
    asteroid = scenario_mascon.grav_optimizer.asteroid
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
            sc_orbits[i][j].data.dt_sample = 1
            sc_orbits[i][j].propagate(oe0, tf)

    # Plot orbits
    plot_orb(sc_orbits)
    plt.show()
