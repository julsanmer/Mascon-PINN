import numpy as np
import pickle as pck
import os
from timeit import default_timer as timer

import src.gravRegression.mascon.initializers as place_positions
from src.gravRegression.gravityOptimizer import GravityOptimizer
from src.celestialBodies.shapeModels.polyhedronShape import PolyhedronShape

from Basilisk.fswAlgorithms import masconFit


# Mascon distribution
class MasconOptimizer(GravityOptimizer):
    def __init__(self, mu_M=None, xyz_M=None):
        super().__init__()

        # Set model name
        self.name = 'mascon'

        # Set mascon parameters
        self.mu_M = mu_M
        self.xyz_M = xyz_M
        if self.mu_M is not None:
            self.n_M = len(mu_M)
        else:
            self.n_M = []

        # Preallocate training time
        self.t_train = []

        # Declare mascon optimizer
        self.optimizer = None

        # Shape and its attributes for
        # interior constraint
        self.shape = []
        self.file_poly = []

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
        shape.xyzVertex = self.shape.xyz_vert.tolist()
        shape.orderFacet = self.shape.order_face.tolist()
        shape.initializeParameters()

        # Initialize mascon distribution
        self.optimizer.muM = self.mu_M.tolist()
        self.optimizer.xyzM = self.xyz_M.tolist()

    # This method trains mascon distribution
    def optimize(self, pos_data, acc_data):
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
        # Set optimizer and shape object to none
        self.shape = None
        self.optimizer = None

    # This method adds a polyhedron shape as
    # interior constraint for mascons
    def add_poly_shape(self, file_poly):
        # Set polyhedron file and instantiate shape
        self.file_poly = file_poly
        self.shape = PolyhedronShape(self.file_poly)

    # This method initializes mascon distribution
    def init_distribution(self, n_M, mu, init='octant', seed=0):
        # Set initialization randomness
        np.random.seed(seed)

        # Do initialization
        if init == 'full':
            xyz_M = place_positions.full_initializer(self.shape, n_M)
        elif init == 'octant':
            xyz_M = place_positions.octant_initializer(self.shape, n_M)
        elif init == 'surface':
            xyz_M = place_positions.surface_initializer(self.shape, n_M)

        # Set mascon distribution output
        mu_M = np.ones(n_M) * mu / (n_M+1)
        self.mu_M = np.concatenate(([mu - np.sum(mu_M)],
                                    mu_M))
        self.xyz_M = np.concatenate((np.array([[0, 0, 0]]),
                                     xyz_M), axis=0)
        self.n_M = len(self.xyz_M)

    # Save regressed model
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

        mascon_dict = {'mu_M': self.mu_M,
                       'xyz_M': self.xyz_M}

        # Save simulation outputs
        with open(file_mascon, "wb") as f:
            pck.dump(mascon_dict, f)
