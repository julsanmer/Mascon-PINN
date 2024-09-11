import numpy as np

from src.groundtruth.groundtruth import Groundtruth
from src.gravRegression.mascon.masconOptimizer import MasconOptimizer
from src.gravRegression.pinn.pinnOptimizer import PINNOptimizer

from src.celestialBodies.asteroid import Asteroid
from src.orbiters.spacecraft import Spacecraft
from src.gravMaps.gravityMaps import GravityMap

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the main scenario class
class Scenario:
    def __init__(self, config):
        # Groundtruth subclass
        self.groundtruth = Groundtruth()

        # Gravity regression attributes
        self.grav_optimizer = None

        # Saves configuration
        self.config = config

    # This method initializes groundtruth
    def init_groundtruth(self):
        # Retrieve groundtruth and its config
        gt = self.groundtruth
        config_gt = self.config['groundtruth']

        # Create groundtruth file
        gt.set_file(config_gt)

        # Initialize asteroid and spacecraft objects
        gt.asteroid = Asteroid()
        gt.asteroid.set_properties(config_gt['asteroid_name'],
                                   file_shape=config_gt['file_poly'])
        gt.spacecraft = Spacecraft()

        # Dense dataset is defined by distribution type,
        # number of data and maximum radius
        gt.data_type = 'dense'
        gt.dense_type = config_gt['dense']['dist']
        gt.n_data = config_gt['dense']['n_data']
        gt.rmax_dense = config_gt['dense']['rmax']

        # We'll add ejecta for low altitude samples
        gt.ejecta = Spacecraft()
        gt.ejecta_type = config_gt['ejecta']['dist']
        gt.n_ejecta = config_gt['ejecta']['n_data']
        gt.rmax_ejecta = config_gt['ejecta']['rmax']

        # Initialize gravity maps
        config_gravmap = config_gt['gravmap']
        gt.gravmap = GravityMap(config_gravmap['rmax_2D'],
                                config_gravmap['rmax_3D'],
                                n_2D=config_gravmap['n_2D'],
                                nr_3D=config_gravmap['nr_3D'],
                                nlat_3D=config_gravmap['nlat_3D'],
                                nlon_3D=config_gravmap['nlon_3D'])

    # This method initializes gravity regression
    def init_regression(self):
        # Declare optimizer, asteroid and gravity map
        if self.config['regression']['grav_model'] == 'mascon':
            self.grav_optimizer = MasconOptimizer()
        elif self.config['regression']['grav_model'] == 'pinn':
            self.grav_optimizer = PINNOptimizer()

        # Declare asteroid and gravity map
        self.grav_optimizer.asteroid = Asteroid()
        self.grav_optimizer.gravmap = GravityMap()

        # Set file
        self.grav_optimizer.set_file(self.config)

        # Declare asteroid under consideration
        config_gt = self.config['groundtruth']
        self.grav_optimizer.asteroid.set_properties(config_gt['asteroid_name'],
                                                    file_shape=config_gt['file_poly'])
