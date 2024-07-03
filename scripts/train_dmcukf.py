import numpy as np

from src.classes.inner.measurements import Measurements
from src.classes.scenario import Scenario
from plots.plots_dmcukf import all_dmcukfplots

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This sets a configuration dict for dmc-ufk
def configuration():
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
                                    'n_2D': 160, 'rmax_2D': 60 * km2m}},
        'measurements': {'file': '',  # filled automatically
                         'dt_sample': 1,
                         'f': 25 * 1e-3,
                         'n_lmk': 100,
                         'dev_lmk': 0,
                         'mask_sun': -90 * np.pi/180},
        'results': {'file': '',
                    'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                    'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                    'dmc_ukf': {'dt_sample': 60,
                                'orbits': np.linspace(0, 10, 11)},
                    'data': {'dmcukf_rate': 60,
                             'grav_rate': 60,
                             'add_ejecta': False,  # bool
                             'n_ejecta': 50,
                             'dev_ejecta': 0,
                             'n_data': 10000},
                    'grad_descent': {'maxiter': 5000,
                                     'lr': 1e-3,
                                     'loss': 'MSE',  # 'MSE' / 'MLE'
                                     'train_xyz': True},
                    'mascon': {'train_xyz': True,  # bool
                               'init': 'octant',
                               'n_M': 100,
                               'seed_M': 0}},
    }

    return config


if __name__ == "__main__":
    # Get configuration
    config = configuration()
    config_dmcukf = config['results']['dmc_ukf']

    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    groundtruth = scenario.groundtruth
    scenario.set_file_groundtruth()
    groundtruth.import_data(dt=config_dmcukf['dt_sample'])

    # Import measurements
    scenario.measurements = Measurements()
    measurements = scenario.measurements
    scenario.set_file_measurements()
    measurements.import_data(dt=config_dmcukf['dt_sample'])

    # Initialize dmc-ukf
    scenario.init_dmcukf()
    dmcukf = scenario.dmcukf

    # Loop through orbits
    t = groundtruth.spacecraft.data.t
    n_orbits = len(config_dmcukf['orbits']) - 1
    T_orb = 2*np.pi / \
            np.sqrt(groundtruth.asteroid.mu/groundtruth.spacecraft.oe[0]**3)
    for i in range(n_orbits):
        # Set initial and final time
        t0 = config_dmcukf['orbits'][i] * T_orb
        tf = config_dmcukf['orbits'][i+1] * T_orb

        # Extract corresponding batch of measurements
        idx = np.where(np.logical_and(t >= t0, t <= tf))[0]
        if i > 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        t_meas = t[idx].tolist()

        # Extract small body position and orientation
        pos_PS_N1 = groundtruth.asteroid.data.pos_PS_N1[idx, :].tolist()
        mrp_PN0 = groundtruth.asteroid.data.mrp_PN0[idx, :].tolist()
        mrp_BP = groundtruth.spacecraft.data.mrp_BP[idx, :].tolist()

        # Extract camera variables
        pixel_lmk = measurements.camera.data.pixel_lmk[idx, :].tolist()
        isvisible_lmk = measurements.camera.data.isvisible_lmk[idx, :].tolist()
        mrp_CP = measurements.camera.data.mrp_CP[idx, :].tolist()

        #
        dmcukf.load_ephemeris(pos_PS_N1, mrp_PN0, mrp_BP)
        dmcukf.load_camera_meas(t_meas, isvisible_lmk, pixel_lmk)

        # Run filter forward
        dmcukf.run_forward()

    all_dmcukfplots(scenario)
