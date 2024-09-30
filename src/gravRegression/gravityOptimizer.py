import numpy as np
import os

from src.celestialBodies.asteroid import Asteroid
from src.orbiters.spacecraft import Spacecraft
from src.gravMaps.gravityMaps import GravityMap

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the gravity regression class
class GravityOptimizer:
    def __init__(self):
        # File
        self.file = []
        self.config = []

        # Type of data
        self.data_type = []

        # Data-related variables (dense)
        self.dense_type = []
        self.n_data = []
        self.rmax_dense = []

        # Define regressed asteroid and
        # gravity map
        self.asteroid = None
        self.gravmap = None

    # This method initializes optimizer
    def prepare_optimizer(self):
        pass

    # This method trains gravity
    def optimize(self, pos_data, acc_data):
        pass

    # This method deletes optimizer
    def delete_optimizer(self):
        pass

    # This method initializes gravity optimizer based
    # on a configuration dictionary
    def initialize(self, config_gt, config_regression):
        # Save configuration
        self.config = config_regression

        # Declare asteroid and gravity map
        self.asteroid = Asteroid()
        self.gravmap = GravityMap()

        # Set file
        self.set_file(config_gt, config_regression)

        # Declare asteroid under consideration
        self.config = config_regression
        self.asteroid.set_properties(config_gt['asteroid_name'],
                                     file_shape=config_gt['file_poly'])

    # This method sets the file for gravity regression
    def set_file(self, config_gt, config_regression):
        # This adds results path
        def _pathgroundtruth(file_path):
            # Create asteroid folder if it does not exist
            exist = os.path.exists(file_path)
            if not exist:
                os.makedirs(file_path)

            # Retrieve groundtruth parameters
            grav_groundtruth = config_gt['grav_model']
            if config_gt['mascon']['add']:
                grav_groundtruth += 'heterogeneous'
            grav_groundtruth += str(config_gt['n_face']) + 'faces'

            # Set results file if required
            file_path += '/regression/' + grav_groundtruth

            return file_path

        # This adds regression attributes
        def _pathregression(file_path):
            # Data and loss
            data = config_gt['data']

            # Define visibility
            if data == 'dense':
                ejecta_str = ''
                visibility_str = ''

            # Define results path
            if data == 'dense':
                type = config_gt['dense']['dist']
                rmax = config_gt['dense']['rmax'] / 1e3
                n_data = config_regression['data']['n_data']

                file_path += '/ideal' + '/dense_' + type \
                             + str(int(rmax)) + 'km_' + str(n_data) + 'samples'

            # Define results path
            exist = os.path.exists(file_path)
            if not exist:
                os.makedirs(file_path)

            return file_path

        # This identifies model
        def _addmodel(file_path):
            ejecta_str = ''
            visibility_str = ''

            # If model is mascon
            if grav_results == 'mascon':
                # Retrieve loss
                loss = config_regression['grad_descent']['loss']

                # Retrieve mascon parameters
                config_mascon = config_regression['mascon']
                n_M = config_mascon['n_M']
                init = config_mascon['init']
                seed_M = config_mascon['seed_M']
                if config_mascon['train_xyz']:
                    mascon_train = 'muxyz'
                else:
                    mascon_train = 'mu'

                # Set mascon results file
                file_path += '/mascon' + str(n_M) + '_' \
                             + mascon_train + '_' + loss + '_' + init + visibility_str \
                             + ejecta_str + 'rand' + str(seed_M) + '.pck'
            # If model is pinn
            elif grav_results == 'pinn':
                # Retrieve loss
                loss = config_regression['grad_descent']['loss']

                # Set bc model
                config_nn = config_regression['nn']
                bc_model = config_nn['model_bc']['model']
                n_M = config_nn['model_bc']['n_M']
                layers = config_nn['layers']
                n_layer = config_nn['neurons']
                activation = config_nn['activation']

                # Set pinn results file
                file_path += '/' + config_nn['model'] + str(layers) + 'x' + str(n_layer) \
                             + activation + '_' + loss + '_' + bc_model \
                             + str(n_M) + visibility_str + ejecta_str + '.pck'
            # If model is spherical harmonics
            elif grav_results == 'spherharm':
                # Set degree
                deg = config_regression['spherharm']['deg']

                # Set spherical harmonics results file
                file_path += '/spherharm_' + str(deg) + 'th.pck'

            return file_path

        # Initiate path with asteroid name
        asteroid_name = config_gt['asteroid_name']
        file_path = 'Results/' + asteroid_name

        # Define Adam gradient descent variables
        grav_results = config_regression['grav_model']

        # Set groundtruth extension
        file_path = _pathgroundtruth(file_path)

        # Set estimation if required
        if grav_results == 'mascon' or grav_results == 'pinn':
            file_path = _pathregression(file_path)

        # Set model
        file_path = _addmodel(file_path)

        # Set results file
        self.file = file_path
