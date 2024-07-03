import numpy as np
import os

from src.spaceObjects.asteroid import Asteroid
from src.spaceObjects.spacecraft import Spacecraft
from src.dmcukf.dmcukf import DMCUKF
from src.gravity.gravityMaps import GravityMap
from src.groundtruth.groundtruth import Groundtruth
from src.gravity.estimation import Estimation
from src.measurements.measurements import Measurements

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the main scenario class
class Scenario:
    def __init__(self, config):
        # Groundtruth subclass
        self.groundtruth = Groundtruth()

        # Measurements subclass
        self.measurements = None

        # DMC-UKF subclass
        self.dmcukf = None

        # Gravity estimation subclass
        self.estimation = None

        # Saves configuration
        self.config = config

    # This method initializes groundtruth
    def init_groundtruth(self):
        # Retrieve groundtruth and its config
        groundtruth = self.groundtruth
        config_gt = self.config['groundtruth']

        # Create groundtruth file
        groundtruth.set_file(config_gt)

        # Initialize asteroid and spacecraft objects
        groundtruth.asteroid = Asteroid()
        groundtruth.asteroid.set_properties(config_gt['asteroid_name'],
                                            file_shape=config_gt['file_poly'])
        groundtruth.spacecraft = Spacecraft()

        # Set type of dataset
        if config_gt['data'] == 'dense':
            # Dense dataset is defined by distribution type,
            # number of data and maximum radius
            groundtruth.data_type = 'dense'
            groundtruth.dense_type = config_gt['dense']['dist']
            groundtruth.n_data = config_gt['dense']['n_data']
            groundtruth.rmax_dense = config_gt['dense']['rmax']

            # We'll add ejecta for low altitude samples
            groundtruth.ejecta = Spacecraft()
            groundtruth.ejecta_type = config_gt['ejecta']['dist']
            groundtruth.n_ejecta = config_gt['ejecta']['n_data']
            groundtruth.rmax_ejecta = config_gt['ejecta']['rmax']
        elif config_gt['data'] == 'orbit':
            # Orbit dataset is defined by initial orbital elements,
            #
            groundtruth.data_type = 'orbit'
            groundtruth.spacecraft.dt_dyn = config_gt['spacecraft']['dt_dyn']
            groundtruth.spacecraft.data.dt_sample = config_gt['spacecraft']['dt_sample']
            groundtruth.spacecraft.oe = \
                config_gt['spacecraft']['oe_0']
            T_orb = np.sqrt(groundtruth.asteroid.mu
                            / groundtruth.spacecraft.oe[0]**3)
            orb_period = 2*np.pi / T_orb
            groundtruth.t_prop = orb_period * config_gt['spacecraft']['n_orbits']

            # We need to add ejecta as well
            groundtruth.ejecta = Spacecraft()

            # Ejecta dataset is defined by distribution type,
            # number of data and maximum radius
            groundtruth.ejecta_type = config_gt['ejecta']['dist']
            groundtruth.n_ejecta = config_gt['ejecta']['n_data']
            groundtruth.rmax_ejecta = config_gt['ejecta']['rmax']

        # Initialize gravity maps
        config_gravmap = config_gt['gravmap']
        groundtruth.gravmap = GravityMap(config_gravmap['rmax_2D'],
                                         config_gravmap['rmax_3D'],
                                         n_2D=config_gravmap['n_2D'],
                                         nr_3D=config_gravmap['nr_3D'],
                                         nlat_3D=config_gravmap['nlat_3D'],
                                         nlon_3D=config_gravmap['nlon_3D'])

    # This method initializes measurements
    def init_measurements(self):
        # Create measurements object and store it
        self.measurements = Measurements()
        measurements = self.measurements

        # Create measurements file
        self.set_file_measurements()

        # Add camera and create landmarks
        measurements.add_camera(nx_pixel=2048, ny_pixel=1536, f=25*1e-3,
                                w_pixel=(17.3*1e-3)/2048, maskangle_cam=0.,
                                maskangle_sun=0.)
        measurements.camera.create_landmark_distribution(
            self.groundtruth.asteroid, n_lmk=100, dev_lmk=5)

        # Initialize BSK camera
        measurements.camera.initialize()

    # This method initializes gravity estimation
    def init_estimation(self):
        # Create estimation instance and store it
        self.estimation = Estimation()
        estimation = self.estimation

        # Set file
        estimation.set_file(self.config)

        # Create asteroid in estimation
        estimation.asteroid = Asteroid()
        estimation.asteroid.set_properties(self.config['groundtruth']['asteroid_name'],
                                           file_shape=self.config['groundtruth']['file_poly'])

        # Initialize gravity maps
        estimation.gravmap = GravityMap()

    # This method initializes dmc-ukf
    def init_dmcukf(self):
        self.dmcukf = DMCUKF()
        dmcukf = self.dmcukf

        dmcukf.create_dmcukf(self.groundtruth.asteroid,
                             self.groundtruth.spacecraft,
                             self.measurements)

    # This sets a file for measurements
    def set_file_measurements(self):
        # Retrieve groundtruth and meas config
        config_gt = self.config['groundtruth']
        config_meas = self.config['measurements']

        # Obtain asteroid path
        path_asteroid = self._set_path_asteroid()

        # Collect groundtruth parameters:
        # asteroid gravity model
        grav_gt = config_gt['grav_model']
        if config_gt['mascon']['add']:
            grav_gt += 'heterogeneous'

        # Create asteroid gravity folder if it does not exist
        path_meas = path_asteroid + '/measurements/' + grav_gt
        exist = os.path.exists(path_meas)
        if not exist:
            os.makedirs(path_meas)

        # Orbit dataset is defined by semi-major axis,
        # inclination and number of orbits
        a0 = config_gt['spacecraft']['oe_0'][0]
        inc0 = config_gt['spacecraft']['oe_0'][2]
        n_orbits = config_gt['spacecraft']['n_orbits']

        # Measurements are defined by focal length, number of
        # landmarks and lighting
        f = config_meas['f']
        n_lmk = config_meas['n_lmk']
        mask_sun = config_meas['mask_sun']
        if mask_sun >= 0:
            lighting_str = 'shadow'
        else:
            lighting_str = 'visible'

        # Create camera path if it does not exist and define camera file
        path_meas2 = path_meas + '/a' + str(int(a0 / 1e3)) + 'km' \
                     + 'i' + str(int(inc0 * 180 / np.pi)) + 'deg' \
                     + '_' + str(n_orbits) + 'orbits'
        exist = os.path.exists(path_meas2)
        if not exist:
            os.makedirs(path_meas2)

        # Define measurements file
        file_meas = path_meas2 + '/' + str(int(f * 1e3)) + 'mm' + '_' \
                    + str(n_lmk) + 'landmarks' + '_' + lighting_str + '.pck'

        # Set measurements file in its class
        self.measurements.file = file_meas