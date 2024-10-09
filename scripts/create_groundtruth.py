import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os

from src.groundtruth.groundtruth import Groundtruth
from plots.plots_groundtruth import all_groundtruth_plots

# Import current directory
current_dir = os.getcwd()

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This sets groundtruth configuration
def configuration():
    # Set configuration file
    config_gt = {'file': '',  # filled automatically
                 'asteroid_name': 'eros',  # 'eros'
                 'grav_model': 'poly',
                 #'file_poly': current_dir + '/Polyhedron_files/eros/'
                 #             + 'eros200700.tab',
                 'file_poly': current_dir + '/Polyhedron_files/eros/'
                              + 'eros007790.tab',
                 'n_face': [],  # to be filled later
                 'data': 'dense',  # 'dense' / 'orbit'
                 'mascon': {'add': False,
                            'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                            'xyz_M': np.array([[8, 0, 0],
                                               [-8, 0, 0]])*km2m},
                 'dense': {'dist': 'rad',  # 'alt', 'rad', 'ell'
                           'rmax': 50 * km2m,
                           'n_data': 100000},
                 'ejecta': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                            'rmax': 1e-3 * km2m,
                            'n_data': 100000},
                 'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160*km2m,
                             'n_2D': 160, 'rmax_2D': 60*km2m}}

    return config_gt


if __name__ == "__main__":
    # Generate configuration
    config_gt = configuration()

    # Create groundtruth instance
    # and initialize
    gt = Groundtruth()
    gt.initialize(config_gt)

    # Set asteroid gravity properties
    asteroid = gt.asteroid
    asteroid.add_poly(config_gt['file_poly'])
    if config_gt['mascon']['add']:
        asteroid.add_mascon(config_gt['mascon']['mu_M'],
                            config_gt['mascon']['xyz_M'])

    # Generate groundtruth data
    gt.generate_data()

    # Create gravity grids and maps
    gt.gravmap.create_grids(asteroid.shape)
    gt.gravmap.generate_maps(asteroid)

    # Delete asteroid swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(gt.file, "wb") as f:
        pck.dump(gt, f)

    # Plot groundtruth plots
    all_groundtruth_plots(gt)
