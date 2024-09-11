import numpy as np
import os

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the gravity regression class
class GravityOptimizer:
    def __init__(self):
        # File
        self.file = []

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

    # This method sets the file for gravity regression
    def set_file(self, config):
        # This adds results path
        def _pathgroundtruth(file_path):
            # Create asteroid folder if it does not exist
            exist = os.path.exists(file_path)
            if not exist:
                os.makedirs(file_path)

            # Retrieve groundtruth parameters
            grav_groundtruth = config['groundtruth']['grav_model']
            if config['groundtruth']['mascon']['add']:
                grav_groundtruth += 'heterogeneous'
            grav_groundtruth += str(config['groundtruth']['n_face']) + 'faces'

            # Set results file if required
            file_path += '/results/' + grav_groundtruth

            return file_path

        # This adds regression attributes
        def _pathregression(file_path):
            # Data and loss
            data = config['groundtruth']['data']

            # Define visibility
            if data == 'dense':
                ejecta_str = ''
                visibility_str = ''

            # Define results path
            if data == 'dense':
                type = config['groundtruth']['dense']['dist']
                rmax = config['groundtruth']['dense']['rmax'] / 1e3
                n_data = config['regression']['data']['n_data']

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
                loss = config['regression']['grad_descent']['loss']

                # Retrieve mascon parameters
                config_mascon = config['regression']['mascon']
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
                loss = config['regression']['grad_descent']['loss']

                # Set bc model
                config_pinn = config['regression']['pinn']
                bc_model = config_pinn['model_bc']['type']
                n_M = config_pinn['model_bc']['n_M']
                layers = config_pinn['layers']
                n_layer = config_pinn['neurons']
                activation = config_pinn['activation']

                # Set pinn results file
                file_path += '/pinn' + str(layers) + 'x' + str(n_layer) \
                             + activation + '_' + loss + '_' + bc_model \
                             + str(n_M) + visibility_str + ejecta_str + '.pck'
            # If model is spherical harmonics
            elif grav_results == 'spherharm':
                # Set degree
                deg = config['estimation']['spherharm']['deg']

                # Set spherical harmonics results file
                file_path += '/spherharm_' + str(deg) + 'th.pck'

            return file_path

        # Initiate path with asteroid name
        asteroid_name = config['groundtruth']['asteroid_name']
        file_path = 'Results/' + asteroid_name

        # Define Adam gradient descent variables
        grav_results = config['regression']['grav_model']

        # Set groundtruth extension
        file_path = _pathgroundtruth(file_path)

        # Set estimation if required
        if grav_results == 'mascon' or grav_results == 'pinn':
            file_path = _pathregression(file_path)

        # Set model
        file_path = _addmodel(file_path)

        # Set results file
        self.file = file_path
