import numpy as np

from src.celestialBodies.gravityModels.gravity import Gravity

from Basilisk.utilities import simIncludeGravBody


# This is the PINN class
class PINNGrav(Gravity):
    def __init__(self, file_torch):
        super().__init__()

        # Set model name
        self.name = 'pinn'

        # Set model file
        self.file_torch = file_torch

        # Preallocate BSK instance and
        # create gravity
        self.gravity_bsk = None
        self.create_gravity()

    # This method creates PINN gravity
    def create_gravity(self):
        # Create gravity factory and object
        gravFactory = simIncludeGravBody.gravBodyFactory()
        gravity = gravFactory.createCustomGravObject('pinn', mu=0.)

        # Set mascon model and initialize it
        gravity.usePINN2GravityModel()
        self.gravity_bsk = gravity.pinn2
        self.gravity_bsk.PINNPath = self.file_torch
        self.gravity_bsk.initializeParameters()

    # This method computes PINN gravity
    def compute_acc(self, pos):
        # Evaluate pinn
        acc = np.array(self.gravity_bsk.computeField(pos)).reshape(3)

        return acc

    # This method eliminates PINN gravity
    def delete_gravity(self):
        self.gravity_bsk = None
