import numpy as np

from src.celestialBodies.gravityModels.gravity import Gravity

from Basilisk.utilities import simIncludeGravBody


# Mascon distribution
class MasconGrav(Gravity):
    def __init__(self, mu_M, xyz_M):
        super().__init__()

        # Set model name
        self.name = 'mascon'

        # Set mascon parameters
        self.mu_M = mu_M
        self.xyz_M = xyz_M
        self.n_M = len(mu_M)

        # Set total gravity parameter
        self.mu = np.sum(mu_M)

        # Preallocate BSK instance and
        # create gravity
        self.gravity_bsk = None
        self.create_gravity()

    # This method creates mascon model
    def create_gravity(self):
        # Create gravity factory and object
        gravFactory = simIncludeGravBody.gravBodyFactory()
        gravity = gravFactory.createCustomGravObject('mascon', mu=self.mu)

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
