import numpy as np
import os
import pickle as pck

from src.bskObjects.propagator import Propagator
from src.groundtruth.position_sample import alt_sample, \
    ell_sample, rad_sample

from Basilisk.simulation import gravityEffector
from Basilisk import __path__
bsk_path = __path__[0]

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

        # Preallocate asteroid, ejecta
        # and spacecraft objects
        self.asteroid = None
        self.ejecta = None
        self.spacecraft = None

        # Preallocate gravity map
        self.gravmap = None

    # This method generates groundtruth data
    def generate_data(self):
        # Set type of ground truth data creation
        if self.data_type == 'orbit':
            # Create orbit data
            self._orbit_data()

            # Create ejecta data
            self._dense_data(self.ejecta,
                             n_data=self.n_ejecta,
                             rmax=self.rmax_ejecta,
                             type=self.ejecta_type)

            # Save generated data
            #scenario.orbit_data(self)
            #scenario.ejecta_data(self)
        elif self.data_type == 'dense':
            # Create dense data
            self._dense_data(self.spacecraft,
                             n_data=self.n_data,
                             rmax=self.rmax_dense,
                             type=self.dense_type)

            # Create ejecta data (the idea is to have
            # a very low altitude dataset)
            self._dense_data(self.ejecta,
                             n_data=self.n_ejecta,
                             rmax=self.rmax_ejecta,
                             type=self.ejecta_type)

    # This imports groundtruth data
    def import_data(self, dt=None, n_data=None):
        # Import groundtruth data from file
        inputs = pck.load(open(self.file, "rb"))

        # Load groundtruth gravity map, asteroid
        # and spacecraft
        self.gravmap = inputs.groundtruth.gravmap
        self.asteroid = inputs.groundtruth.asteroid
        self.spacecraft = inputs.groundtruth.spacecraft
        self.ejecta = inputs.groundtruth.ejecta

        # Define groundtruth copy
        data_in = inputs.groundtruth

        # Determine indexes to prune data
        if n_data is not None:
            # Number of data and indexes
            idx = np.linspace(0, n_data-1, n_data).astype(int)
        elif dt is not None:
            # Initial, final time, sampling rate and indexes
            t0 = self.spacecraft.data.t[0]
            tf = self.spacecraft.data.t[-1]
            idx = np.linspace(0, np.floor((tf - t0) / dt) * dt,
                              int(np.floor((tf - t0) / dt)) + 1).astype(int)

        # Prune dataset related variables
        self.spacecraft.data.pos_BP_P = data_in.spacecraft.data.pos_BP_P[idx, :]
        self.spacecraft.data.acc_BP_P = data_in.spacecraft.data.acc_BP_P[idx, :]
        self.spacecraft.data.r_BP = data_in.spacecraft.data.r_BP[idx]
        self.spacecraft.data.h_BP = data_in.spacecraft.data.h_BP[idx]
        self.spacecraft.data.U = data_in.spacecraft.data.U[idx]

        self.ejecta.data.pos_BP_P = data_in.ejecta.data.pos_BP_P[idx, :]
        self.ejecta.data.acc_BP_P = data_in.ejecta.data.acc_BP_P[idx, :]
        self.ejecta.data.r_BP = data_in.ejecta.data.r_BP[idx]
        self.ejecta.data.h_BP = data_in.ejecta.data.h_BP[idx]
        self.ejecta.data.U = data_in.ejecta.data.U[idx]

        # If data type is orbit
        if dt is not None:
            # Prune spacecraft time, position and velocity
            self.spacecraft.data.t = data_in.spacecraft.data.t[idx]
            self.spacecraft.data.pos_BP_N0 = data_in.spacecraft.data.pos_BP_N0[idx, :]
            self.spacecraft.data.vel_BP_N0 = data_in.spacecraft.data.vel_BP_N0[idx, :]
            self.spacecraft.data.pos_BP_N1 = data_in.spacecraft.data.pos_BP_N1[idx, :]
            self.spacecraft.data.vel_BP_N1 = data_in.spacecraft.data.vel_BP_N1[idx, :]
            self.spacecraft.data.vel_BP_P = data_in.spacecraft.data.vel_BP_P[idx, :]
            #self.spacecraft.data.accHigh_BP_P = data_in.spacecraft.data.accHigh_BP_P[idx, :]

            # Prune Sun's position
            self.asteroid.data.pos_PS_N1 = data_in.asteroid.data.pos_PS_N1[idx, :]
            self.asteroid.data.e_SP_P = data_in.asteroid.data.e_SP_P[idx, :]

            # Prune spacecraft mrp
            self.asteroid.data.mrp_PN0 = data_in.asteroid.data.mrp_PN0[idx, :]
            self.spacecraft.data.mrp_BP = data_in.spacecraft.data.mrp_BP[idx, :]

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

        # Define groundtruth file depending if data is on-orbit
        # or dense
        if data == 'orbit':
            # Orbit dataset is defined by semi-major axis,
            # inclination and number of orbits
            a0 = config_gt['spacecraft']['oe_0'][0]
            inc0 = config_gt['spacecraft']['oe_0'][2]
            n_orbits = config_gt['spacecraft']['n_orbits']

            # File
            file_gt = path_gt + '/a' + str(int(a0 / 1e3)) + 'km' \
                      + 'i' + str(int(inc0 * 180/np.pi)) + 'deg' + '_' \
                      + str(n_orbits) + 'orbits' + '.pck'
        elif data == 'dense':
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

    # This internal method creates dense data
    # around the asteroid
    def _dense_data(self, space_object, n_data=1000,
                    rmax=None, type='alt'):
        # Get gravity and shape objects
        asteroid = self.asteroid
        shape = asteroid.shape

        # Preallocate dataset
        n = n_data
        pos_BP_P, acc_BP_P = \
            np.zeros((n, 3)), np.zeros((n, 3))
        r_BP, h_BP = \
            np.zeros(n), np.zeros(n)
        U = np.zeros(n)

        # Set type of dense data
        if type == 'alt':
            # Set maximum altitude
            hmax = rmax
            n_face = asteroid.shape.n_face
            xyz_face = asteroid.shape.xyz_face

        # Create position-gravity acceleration time
        print('------- Initiating ground truth -------')
        np.random.seed(0)
        for i in range(n):
            # Set exterior point flag to false
            flag_ext = False
            while not flag_ext:
                if type == 'rad':
                    pos = rad_sample(rmax)
                elif type == 'ell':
                    pos = ell_sample(rmax, axes)
                elif type == 'alt':
                    pos = alt_sample(hmax, xyz_face)

                # Check if point is exterior
                is_exterior = shape.check_exterior(pos.tolist())

                # Evaluate only if it is an exterior point
                if is_exterior:
                    # Compute gravity and potential
                    pos_BP_P[i, 0:3] = pos
                    acc_BP_P[i, 0:3] = asteroid.compute_gravity(pos)
                    U[i] = asteroid.compute_potential(pos)

                    # Compute radius and altitude
                    r_BP[i] = np.linalg.norm(pos)
                    h_BP[i] = asteroid.shape.compute_altitude(pos)

                    # Set flag to exit
                    flag_ext = True

            # Print status
            if (i+1) % int(n/100) == 0:
                print(str(int((i+1)/n * 100)) + ' % data generated')

        # Output status
        print('------- Finished ground truth -------')

        # Save generated dataset
        data = space_object.data
        data.pos_BP_P = pos_BP_P
        data.acc_BP_P = acc_BP_P
        data.U = U
        data.r_BP = r_BP
        data.h_BP = h_BP

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
