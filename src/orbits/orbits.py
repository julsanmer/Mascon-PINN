import numpy as np
import pickle as pck
import os
from timeit import default_timer as timer

from src.bskObjects.propagator import Propagator
from src.spaceObjects.asteroid import Asteroid
from src.spaceObjects.spacecraft import Spacecraft

from Basilisk.simulation import gravityEffector
from Basilisk import __path__
bsk_path = __path__[0]

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the orbits class
class Orbits:
    def __init__(self, config):
        # Set an asteroid
        self.asteroid = Asteroid()

        # Initialize propagator
        self.propagator = None

        # Set propagation time
        self.t_prop = config['t_prop']

        # Get initial orbital elements
        a = config['oe']['a']
        ecc = config['oe']['ecc']
        inc = config['oe']['inc']
        RAAN = config['oe']['RAAN']
        omega = config['oe']['omega']
        f = config['oe']['f']

        # Get number of different a0 and inc0
        self.n_a = len(a)
        self.n_inc = len(inc)

        # Create a 2D list to store different orbits
        self.orbits = []

        # Create an array to store computational time
        self.t_cpu = np.zeros((self.n_a, self.n_inc))

        # Loop through semi-major axis
        for i in range(self.n_a):
            # Empty rows
            row = []

            # Loop through inclination
            for j in range(self.n_inc):
                # Create spacecraft instance
                oe = np.array([a[i],
                               ecc,
                               inc[j],
                               RAAN,
                               omega,
                               f])
                orbit = Spacecraft(oe=oe)
                orbit.data.dt_sample = config['groundtruth']['dt_sample']
                row.append(orbit)

            # Append the row to the 2D list
            self.orbits.append(row)

    # Propagate different orbits
    def propagate(self):
        print('------- Initiating propagation -------')

        # Loop semi-major axis
        for i in range(self.n_a):
            # Loop inclination
            for j in range(self.n_inc):
                # Create propagator instance and initialize
                propagator = Propagator(self.asteroid,
                                        self.orbits[i][j])
                propagator.init_sim()

                # Start measuring cpu time
                t_start = timer()

                # Propagate
                propagator.simulate(self.t_prop)

                # End measuring cpu time
                t_end = timer()
                self.t_cpu = t_end - t_start

                # Save data
                propagator.save_outputs(self.asteroid,
                                        self.orbits[i][j])

                # Print status
                print(str(i+1) + '/' + str(self.n_a) + ', ' +
                      str(j+1) + '/' + str(self.n_inc)
                      + ' orbits completed')

        # Print finish
        print('------- Finished propagation -------')

        # Delete swigpy objects
        self.asteroid.delete_swigpy()

    # Compute propagation errors
    def compute_errors(self, file_ref):
        # Load reference
        reference = pck.load(open(file_ref, "rb"))
        orbits_ref = reference.orbits

        # Loop through a0 and i0
        for i in range(self.n_a):
            for j in range(self.n_inc):
                # Get reference trajectory
                pos_ref = orbits_ref[i][j].data.pos_BP_P
                pos = self.orbits[i][j].data.pos_BP_P

                # Compute position error
                pos_err = np.linalg.norm(pos - pos_ref, axis=1)

                # Compute errors
                self.orbits[i][j].data.pos_err = pos_err

    # Sets groundtruth file
    @staticmethod
    def set_filegroundtruth(config_gt):
        # Set asteroid name and groundtruth gravity
        asteroid_name = config_gt['asteroid_name']

        # Create asteroid folder if it does not exist
        path_asteroid = 'Results/' + asteroid_name
        exist = os.path.exists(path_asteroid)
        if not exist:
            os.makedirs(path_asteroid)

        # Retrieve groundtruth parameters
        grav_gt = config_gt['grav_model']
        if config_gt['mascon']['add']:
            grav_gt += 'heterogeneous'

        # Obtain number of faces
        _, _, _, n_face = \
            gravityEffector.loadPolyFromFileToList(config_gt['file_poly'])
        config_gt['n_face'] = n_face

        # Create ground truth path if it does not exist and define ground truth file
        path = path_asteroid + '/groundtruth/' + grav_gt + str(n_face) + 'faces'
        exist = os.path.exists(path)
        if not exist:
            os.makedirs(path)
        file = path + '/propagation.pck'

        return file

    # Sets results file
    @staticmethod
    def set_fileresults(config):
        # Set asteroid name and groundtruth gravity
        asteroid_name = config['groundtruth']['asteroid_name']

        # Create asteroid folder if it does not exist
        path_asteroid = 'Results/' + asteroid_name
        exist = os.path.exists(path_asteroid)
        if not exist:
            os.makedirs(path_asteroid)

        # Retrieve groundtruth parameters
        grav_groundtruth = config['groundtruth']['grav_model']
        if config['groundtruth']['mascon']['add']:
            grav_groundtruth += 'heterogeneous'

        # Set results file
        path = path_asteroid + '/results/' + grav_groundtruth
        file = path + config['estimation']['model_path'] \
               + config['estimation']['file'] + '_orbits.pck'
        file_model = path + config['estimation']['model_path'] \
                     + config['estimation']['file'] + '.pck'

        return file, file_model
