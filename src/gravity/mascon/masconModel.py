import numpy as np
import pickle as pck
import os
from timeit import default_timer as timer

from src.gravity.gravityModel import Gravity
import src.gravity.mascon.initializers as place_positions

from Basilisk.fswAlgorithms import masconFit
from Basilisk.simulation import gravityEffector
from Basilisk.utilities import simIncludeGravBody
from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'


# Mascon distribution
class MasconGrav(Gravity):
    def __init__(self, mu_M=None, xyz_M=None, training=False):
        super().__init__()

        # Set model name
        self.name = 'mascon'

        # Set mascon parameters
        self.mu_M = mu_M
        self.xyz_M = xyz_M
        if self.mu_M is not None:
            self.n_M = len(mu_M)

        # Set total gravity parameter
        self.mu = []

        # Number of masses, seed and
        # non-dimensional parameters
        self.n_M = []

        # Set polyhedron properties for interior constraint
        self.polyFile = bsk_path + 'EROS856Vert1708Fac.txt'
        #self.polyFile = bsk_path + 'ver128q.tab'
        #self.polyFile = bsk_path + 'eros007790.tab'
        vert_list, face_list, n_vert, n_face = \
            gravityEffector.loadPolyFromFileToList(self.polyFile)
        self.xyz_vert = np.array(vert_list)
        self.order_face = np.array(face_list)
        self.n_vert = n_vert
        self.n_face = n_face
        self.xyz_face = np.zeros((n_face, 3))
        for i in range(n_face):
            idx = self.order_face[i, 0:3]
            self.xyz_face[i, 0:3] = (self.xyz_vert[idx[0]-1, 0:3]
                                     + self.xyz_vert[idx[1]-1, 0:3]
                                     + self.xyz_vert[idx[2]-1, 0:3]) / 3

        # Preallocate training time
        self.t_train = []

        # Preallocate mascon optimizer
        # and gravity
        self.optimizer = None
        self.gravity_bsk = None

        if not training:
            self.create_gravity()

    # This method prepares mascon optimizer
    def prepare_optimizer(self, loss='quadratic',
                          maxiter=1000,
                          lr=1e-3,
                          batch_size=2000,
                          xyzM_ad=None,
                          muM_ad=None,
                          train_xyz=True):
        # Initialize mascon fit module
        self.optimizer = masconFit.MasconFit()

        # Set adimensional variables
        self.optimizer.mu = np.sum(self.mu_M)
        self.optimizer.muMad = muM_ad
        self.optimizer.xyzMad = xyzM_ad.tolist()

        # Choose algorithm and loss function
        self.optimizer.lossType = loss

        # Set training variables flag
        if train_xyz:
            self.optimizer.trainXYZ = True
        else:
            self.optimizer.trainXYZ = False

        # Set Adam parameters
        self.optimizer.setMaxIter(maxiter)
        self.optimizer.setLR(lr)
        self.optimizer.setBatchSize(batch_size)

        # Set shape polyhedron
        shape = self.optimizer.shapeModel
        shape.xyzVertex = self.xyz_vert.tolist()
        shape.orderFacet = self.order_face.tolist()
        shape.initializeParameters()

        # Initialize mascon distribution
        self.optimizer.muM = self.mu_M.tolist()
        self.optimizer.xyzM = self.xyz_M.tolist()

    # This method trains mascon distribution
    def train(self, pos_data, acc_data):
        # Do optimizer
        if self.n_M > 1:
            # Start measuring cpu time
            t_start = timer()

            # Call optimizer
            self.optimizer.train(pos_data.tolist(),
                                 acc_data.tolist(),
                                 True)

            # End measuring cpu time
            t_end = timer()
            self.t_train.append(t_end - t_start)

        # Save trained distribution
        self.mu_M = np.array(self.optimizer.muM)
        self.xyz_M = np.array(self.optimizer.xyzM)

    # This method deletes mascon optimizer
    def delete_optimizer(self):
        # Set optimizer object to none
        self.optimizer = None

    # This method creates mascon model
    def create_gravity(self):
        # Compute mu
        mu = np.sum(self.mu_M)

        # Create gravity factory and object
        gravFactory = simIncludeGravBody.gravBodyFactory()
        gravity = gravFactory.createCustomGravObject('mascon', mu=mu)

        # Set mascon model and initialize it
        gravity.useMasconGravityModel()
        self.gravity_bsk = gravity.mascon
        self.gravity_bsk.muMascon = self.mu_M.tolist()
        self.gravity_bsk.xyzMascon = self.xyz_M.tolist()
        self.gravity_bsk.initializeParameters()

    # This method computes mascon gravity
    def compute_acc(self, pos):
        # Evaluate gravity
        acc = np.array(self.gravity_bsk.computeField(pos)).reshape(3)

        return acc

    # This method computes mascon potential
    def compute_U(self, pos):
        # Evaluate potential
        U = np.array(self.gravity_bsk.computePotentialEnergy(pos))

        return U

    # This method deletes mascon model
    def delete_gravity(self):
        self.gravity_bsk = None

    # This method initializes mascon distribution
    def init_distribution(self, asteroid, n_M, init='octant', seed=0):
        # Get shape and mu
        shape = asteroid.shape
        mu = asteroid.mu

        # Set initialization randomness
        np.random.seed(seed)

        # Do initialization
        if init == 'full':
            xyz_M = place_positions.full_initializer(shape, n_M)
        elif init == 'octant':
            xyz_M = place_positions.octant_initializer(shape, n_M)
        elif init == 'surface':
            xyz_M = place_positions.surface_initializer(shape, n_M)

        # Set mascon distribution output
        mu_M = np.ones(n_M) * mu / (n_M+1)
        self.mu_M = np.concatenate(([mu - np.sum(mu_M)],
                                    mu_M))
        self.xyz_M = np.concatenate((np.array([[0, 0, 0]]),
                                     xyz_M), axis=0)
        self.n_M = len(self.xyz_M)

    def save_model(self, path):
        # Set path to save
        parts = path.split('/')
        path_mascon = '/'.join(parts[:-1])
        path_masconmodel = path_mascon + '/mascon_models'

        # Create path if it does not exist
        exist = os.path.exists(path_masconmodel)
        if not exist:
            os.makedirs(path_masconmodel)
        model_name = parts[-1].split('.')[0]

        file_mascon = path_masconmodel + '/' + model_name + '.pck'

        mascon_dict = {
            'mu_M': self.mu_M,
            'xyz_M': self.xyz_M
        }

        # Save simulation outputs
        with open(file_mascon, "wb") as f:
            pck.dump(mascon_dict, f)
