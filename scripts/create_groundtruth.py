import matplotlib.pyplot as plt
import numpy as np
import pickle as pck

from src.scenario import Scenario
from plots.plots_gravity import plot_dataset

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This sets groundtruth configuration
def configuration(data):
    # Dense dataset
    if data == 'dense':
        # Set configuration file
        config = {
            'groundtruth': {'file': '',  # filled automatically
                            'asteroid_name': 'eros',  # 'eros'
                            'grav_model': 'poly',
                            #'file_poly': bsk_path + 'eros200700.tab',
                            'file_poly': bsk_path + 'eros007790.tab',
                            'n_face': [],  # to be filled later
                            'data': 'dense',  # 'dense' / 'orbit'
                            'mascon': {'add': False,
                                       'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                                       'xyz_M': np.array([[10, 0, 0],
                                                          [-10, 0, 0]]) * 1e3},
                            'dense': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                                      'rmax': 50 * km2m,
                                      'n_data': 100000},
                            'ejecta': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                                       'rmax': 1e-3 * km2m,
                                       'n_data': 200000},
                            'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160*km2m,
                                        'n_2D': 160, 'rmax_2D': 60*km2m}}
        }
    # On-orbit dataset
    elif data == 'orbit':
        # Set configuration file
        config = {
            'groundtruth': {'file': '',  # filled automatically
                            'asteroid_name': 'eros',  # 'eros'
                            'grav_model': 'poly',  # 'poly'
                            'file_poly': bsk_path + 'eros007790.tab',
                            'data': 'orbit',  # 'dense' / 'orbit'
                            'mascon': {'add': False,
                                       'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                                       'xyz_M': np.array([[10, 0, 0],
                                                          [-10, 0, 0]]) * 1e3},
                            'spacecraft': {'dt_dyn': 1,
                                           'dt_sample': 1,
                                           'n_orbits': 10,
                                           'oe_0': np.array([34 * km2m,  # semi-major axis
                                                             1e-3,  # eccentricity
                                                             45 * deg2rad,  # inclination
                                                             48.2 * deg2rad,  # RAAN
                                                             347.8 * deg2rad,  # periapsis
                                                             85.3 * deg2rad])},  # f
                            'ejecta': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                                       'rmax': 5 * km2m,
                                       'n_data': 1000},
                            'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                                        'n_2D': 160, 'rmax_2D': 60 * km2m}}
        }

    return config


if __name__ == "__main__":
    # Generate configuration
    config = configuration('dense')
    #config = configuration('dense')
    config_gt = config['groundtruth']

    # Create scenario instance
    # and initialize groundtruth
    scenario = Scenario(config)
    scenario.init_groundtruth()
    groundtruth = scenario.groundtruth

    # Set asteroid gravity properties
    asteroid = groundtruth.asteroid
    asteroid.add_poly(config_gt['file_poly'])
    if config_gt['mascon']['add']:
        asteroid.add_mascon(config_gt['mascon']['mu_M'],
                            config_gt['mascon']['xyz_M'])

    # Generate groundtruth data
    groundtruth.generate_data()

    # Create gravity grids and maps
    groundtruth.gravmap.create_grids(asteroid.shape)
    groundtruth.gravmap.generate_maps(asteroid)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(groundtruth.file, "wb") as f:
        pck.dump(scenario, f)

    # Plot dataset
    plot_dataset(groundtruth.spacecraft.data.pos_BP_P,
                 asteroid.shape.xyz_vert,
                 asteroid.shape.order_face)
    plot_dataset(groundtruth.ejecta.data.pos_BP_P,
                 asteroid.shape.xyz_vert,
                 asteroid.shape.order_face)
    plt.show()
