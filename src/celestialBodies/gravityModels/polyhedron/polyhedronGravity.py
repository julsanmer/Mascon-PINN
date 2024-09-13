import numpy as np

from src.celestialBodies.gravityModels.gravity import Gravity

from Basilisk.utilities import simIncludeGravBody

km2m = 1e3


# Polyhedron
class PolyhedronGrav(Gravity):
    def __init__(self, file, mu=1):
        super().__init__()

        # Set model name
        self.name = 'poly'

        # Set standard gravity and
        # polyhedron file
        self.mu = mu
        self.file = file

        # Preallocate BSK instance and
        # create gravity
        self.gravity_bsk = None
        self.create_gravity()

    # This method creates gravity model
    def create_gravity(self):
        # Call gravity factory
        gravFactory = simIncludeGravBody.gravBodyFactory()

        # Create polyhedron model
        gravity = gravFactory.createCustomGravObject('poly', mu=self.mu)
        gravity.usePolyhedralGravityModel(self.file)

        # Set polyhedron model and initialize it
        self.gravity_bsk = gravity.poly
        self.gravity_bsk.muBody = self.mu
        self.gravity_bsk.initializeParameters()

    # This method evaluates gravity
    def compute_acc(self, pos):
        # Evaluate polyhedron
        acc = np.array(self.gravity_bsk.computeField(pos)).reshape(3)

        return acc

    # This method evaluates potential
    def compute_U(self, pos):
        # Evaluate polyhedron
        U = self.gravity_bsk.computePotentialEnergy(pos)

        return U

    # This method eliminates gravity object
    # (important to save)
    def delete_gravity(self):
        self.gravity_bsk = None
