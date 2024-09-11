import numpy as np

from src.gravity.gravityModel import Gravity
from src.gravity.spherharm.spherharm_features import compute_CS

from Basilisk.simulation import gravityEffector
from Basilisk.utilities import simIncludeGravBody


# Spherical harmonics
class SpherharmGrav(Gravity):
    def __init__(self, mu, deg, rE):
        super().__init__()

        # Set name
        self.name = 'spherharm'

        # Set spherharm parameters
        self.mu = mu
        self.rE = rE
        self.deg = deg

        # Preallocate coefficients
        self.C = np.zeros((self.deg+1, self.deg+1))
        self.S = np.zeros((self.deg+1, self.deg+1))

        # BSK gravity model
        self.gravity_bsk = None

    # Polyhedron to spherical harmonics
    def poly2sh(self, file_poly):
        self.C, self.S = compute_CS(self.deg, self.rE, file_poly)

    # This method creates gravity model
    def create_gravity(self):
        # Call gravity factory
        gravFactory = simIncludeGravBody.gravBodyFactory()

        # Create polyhedron model
        gravity = gravFactory.createCustomGravObject('spherharm',
                                                     mu=self.mu)
        gravity.useSphericalHarmParams = True

        # Set spherical harmonics model and initialize it
        self.gravity_bsk = gravity.spherHarm
        self.gravity_bsk.cBar = gravityEffector.MultiArray(self.C)
        self.gravity_bsk.sBar = gravityEffector.MultiArray(self.S)
        self.gravity_bsk.muBody = self.mu
        self.gravity_bsk.maxDeg = self.deg
        self.gravity_bsk.radEquator = self.rE
        self.gravity_bsk.initializeParameters()

    # This method evaluates gravity
    def compute_acc(self, pos):
        # Evaluate polyhedron
        acc = np.array(self.gravity_bsk.computeField(pos)).reshape(3)

        return acc

    # This method eliminates gravity object
    # (important to save)
    def delete_gravity(self):
        self.gravity_bsk = None
