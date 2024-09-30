import numpy as np
import pickle as pck
import os

from src.scenario import Scenario
from src.gravRegression.spherharm.poly_to_spherharm import compute_CS

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
                        'n_face': [],
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
        'regression': {'grav_model': 'spherharm',  # 'mascon' / 'pinn' / 'spherharm'
                       'spherharm': {'deg': 16,
                                     'rE': 16*km2m}}
    }

    return config


if __name__ == "__main__":
    # Generate configuration
    config = configuration()

    C, S = compute_CS(10, 16*km2m, config['groundtruth']['file_poly'])

    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    groundtruth = scenario.groundtruth
    groundtruth.set_file(config['groundtruth'])
    groundtruth.import_data(n_data=config['regression']['data']['n_data'])

    # Initialize estimation
    scenario.init_regression()
    asteroid = scenario.regression.asteroid

    # Add spherical harmonics and compute coefficients
    asteroid.add_spherharm(config['estimation']['spherharm']['deg'],
                           config['estimation']['spherharm']['rE'])
    spherharm = asteroid.gravity[0]
    spherharm.poly2sh(config['groundtruth']['file_poly'])

    # Create spherharm gravity
    spherharm.create_gravity()

    # Import gravity map grids
    gravmap = scenario.estimation.gravmap
    gravmap.import_grids(groundtruth.gravmap)

    # Generate gravity maps and errors
    gravmap.generate_maps(asteroid)
    gravmap.error_maps(groundtruth.gravmap)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(scenario.estimation.file, "wb") as f:
        pck.dump(scenario, f)

    # Plot gravity
    all_gravityplots(scenario)
