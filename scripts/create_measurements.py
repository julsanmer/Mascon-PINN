import numpy as np
import pickle as pck

from src.classes.scenario import Scenario

from Basilisk import __path__
bsk_path = __path__[0]

km2m = 1e3
deg2rad = np.pi/180


# This sets a configuration dict for measurements
def configuration():
    # Set configuration file
    config = {
        'groundtruth': {'file': '',  # filled automatically
                        'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': bsk_path + 'eros007790.tab',
                        'data': 'orbit',  # 'dense' / 'orbit'
                        'mascon': {'add': False,
                                   'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'spacecraft': {'n_orbits': 10,
                                       'oe_0': np.array([34 * km2m,  # semi-major axis
                                                         1e-3,  # eccentricity
                                                         45 * deg2rad,  # inclination
                                                         48.2 * deg2rad,  # RAAN
                                                         347.8 * deg2rad,  # periapsis
                                                         85.3 * deg2rad])},  # f
                        'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                                    'n_2D': 160, 'rmax_2D': 60 * km2m}},
        'measurements': {'file': '',  # filled automatically
                         'dt_sample': 1,
                         'f': 25*1e-3,
                         'n_lmk': 100,
                         'dev_lmk': 0,
                         'mask_sun': -90*np.pi/180}
    }

    return config


if __name__ == "__main__":
    # Get configuration
    config = configuration()
    config_meas = config['measurements']

    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    groundtruth = scenario.groundtruth
    scenario.set_file_groundtruth()
    groundtruth.import_data(dt=config_meas['dt_sample'])

    # Initialize measurements
    scenario.init_measurements()
    measurements = scenario.measurements

    # Create measurements
    measurements.generate_data(groundtruth.asteroid,
                               groundtruth.spacecraft)

    # Delete swigpy
    measurements.camera.delete_swigpy()

    # Save simulation outputs
    with open(measurements.file, "wb") as f:
        pck.dump(scenario, f)
