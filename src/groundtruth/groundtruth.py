import numpy as np
import os
import pickle as pck

from src.groundtruth.dense_dataset import generate_dense_dataset

from Basilisk.simulation import gravityEffector

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# Groundtruth class
class Groundtruth:
    def __init__(self):
        # Groundtruth file
        self.file = []

        self.data_type = []

        # Data-related variables (dense)
        self.dense_type = []
        self.n_data = []
        self.rmax_dense = []

        # Data-related variables (ejecta)
        self.ejecta_type = []
        self.n_ejecta = []
        self.rmax_ejecta = []

        # Declare asteroid, ejecta
        # and spacecraft objects
        self.asteroid = None
        self.ejecta = None
        self.spacecraft = None

        # Declare gravity map
        self.gravmap = None

    # This method generates groundtruth data
    def generate_data(self):
        # Set type of ground truth data creation
        if self.data_type == 'dense':
            # Create dense data
            generate_dense_dataset(self.spacecraft,
                                   self.asteroid,
                                   n_data=self.n_data,
                                   rmax=self.rmax_dense,
                                   type=self.dense_type)

            # Create ejecta data (the idea is to have
            # a very low altitude dataset)
            generate_dense_dataset(self.ejecta,
                                   self.asteroid,
                                   n_data=self.n_ejecta,
                                   rmax=self.rmax_ejecta,
                                   type=self.ejecta_type)

    # This imports groundtruth data
    def import_data(self, n_data=None):
        # Import groundtruth data from file
        inputs = pck.load(open(self.file, "rb"))

        # Load groundtruth gravity map, asteroid
        # and spacecraft
        gt_in = inputs.groundtruth
        self.gravmap = gt_in.gravmap
        self.asteroid = gt_in.asteroid
        self.spacecraft = gt_in.spacecraft
        self.ejecta = gt_in.ejecta

        # Determine indexes to prune data
        if n_data is not None:
            # Number of data and indexes
            idx = np.linspace(0, n_data-1, n_data).astype(int)

        # Spacecraft with pruning
        sc_in = gt_in.spacecraft
        sc_out = self.spacecraft
        sc_out.data.pos_BP_P = sc_in.data.pos_BP_P[idx, :]
        sc_out.data.acc_BP_P = sc_in.data.acc_BP_P[idx, :]
        sc_out.data.r_BP = sc_in.data.r_BP[idx]
        sc_out.data.h_BP = sc_in.data.h_BP[idx]
        sc_out.data.U = sc_in.data.U[idx]

        # Ejecta with pruning
        ej_in = gt_in.ejecta
        ej_out = self.ejecta
        ej_out.data.pos_BP_P = ej_in.data.pos_BP_P[idx, :]
        ej_out.data.acc_BP_P = ej_in.data.acc_BP_P[idx, :]
        ej_out.data.r_BP = ej_in.data.r_BP[idx]
        ej_out.data.h_BP = ej_in.data.h_BP[idx]
        ej_out.data.U = ej_in.data.U[idx]

    # This method sets groundtruth file based on config
    def set_file(self, config_gt):
        # Create asteroid folder if it does not exist
        asteroid_name = config_gt['asteroid_name']
        path_asteroid = 'Results/' + asteroid_name
        exist = os.path.exists(path_asteroid)
        if not exist:
            os.makedirs(path_asteroid)

        # Collect groundtruth parameters: data type
        # and asteroid gravity model
        data = config_gt['data']
        grav_gt = config_gt['grav_model']
        if config_gt['mascon']['add']:
            grav_gt += 'heterogeneous'

        # Obtain number of faces

        _, _, _, n_face = \
            gravityEffector.loadPolyFromFileToList(config_gt['file_poly'])
        config_gt['n_face'] = n_face

        # Create asteroid gravity folder if it does not exist
        path_gt = path_asteroid + '/groundtruth/' + grav_gt + str(n_face) + 'faces'
        exist = os.path.exists(path_gt)
        if not exist:
            os.makedirs(path_gt)

        # Define groundtruth file
        if data == 'dense':
            # Dense dataset is defined by distribution type,
            # number of data and maximum radius
            type = config_gt['dense']['dist']
            rmax = config_gt['dense']['rmax'] / 1e3
            n_data = config_gt['dense']['n_data']

            # File
            file_gt = path_gt + '/dense_' + type \
                      + str(int(rmax)) + 'km_' + str(n_data) \
                      + 'samples' + '.pck'

        # Set groundtruth file in its class
        self.file = file_gt

    # This internal method creates orbit data
    # around the asteroid
    def _orbit_data(self):
        # Create the scenario and initialize
        propagator = Propagator(self.asteroid,
                                self.spacecraft)
        propagator.init_sim()

        # Propagate
        propagator.simulate(self.t_prop)

        # Save data
        propagator.save_outputs(self.asteroid,
                                self.spacecraft)
