import numpy as np

from src.celestialBodies.gravityModels.masconGravity import MasconGrav
from src.celestialBodies.gravityModels.pinnGravity import PINNGrav
from src.celestialBodies.gravityModels.polyhedronGravity import PolyhedronGrav
from src.celestialBodies.shapeModels.polyhedronShape import PolyhedronShape
#from src.celestialBodies.gravityModels. import PINNGrav

from Basilisk.utilities import RigidBodyKinematics as rbk

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the asteroid class
class Asteroid:
    def __init__(self):
        # Asteroid name
        self.name = []

        # Mu, heliocentric orbital elements,
        # orientation and rotation period
        self.mu = []
        self.oe = []
        self.eul313 = []
        self.rot_period = []

        # Heliocentric dcm at t=0
        self.dcm_N1N0 = []

        # Preallocate data
        self.data = self.Data()

        # Preallocate gravity and shape models
        self.gravity_names = []
        self.gravity = []
        self.shape = None

    # This is the inner data class
    class Data:
        def __init__(self):
            # Asteroid data
            self.t = []

            # Position and velocity in
            # heliocentric inertial frame
            self.pos_PS_N0, self.vel_PS_N0 = [], []

            # Position in planet inertial frame
            self.pos_PS_N1 = []

            # Sun's direction in planet
            # rotating frame
            self.e_SP_P = []

            # Euler angles and MRP
            self.eul313_PN0 = []
            self.mrp_PN0 = []
            self.angvel_PN0_P = []

    # This method adds a mascon gravity model
    def add_mascon(self, mu_M=None, xyz_M=None):
        # Set name and append mascon model
        self.gravity_names.append('mascon')
        self.gravity.append(MasconGrav(mu_M=mu_M,
                                       xyz_M=xyz_M))

    # This method adds a pinn gravity model
    def add_pinn(self, file_torch=None):
        # Set name and append pinn model
        self.gravity_names.append('pinn')
        self.gravity.append(PINNGrav(file_torch=file_torch))

    # This method adds a polyhedron gravity model
    def add_poly(self, file_poly):
        # Set name and append polyhedron model
        self.gravity_names.append('poly')
        self.gravity.append(PolyhedronGrav(file_poly, mu=self.mu))

    # # This method adds a SH gravity model
    # def add_spherharm(self, deg, rE):
    #     # Set name and append spherharm model
    #     self.gravity_names.append('spherharm')
    #     self.gravity.append(SpherharmGrav(self.mu, deg, rE))

    # This method computes asteroid gravity acceleration
    def compute_gravity(self, pos):
        # Initialize acceleration
        acc = np.zeros(3)

        # Loop through attached models
        for model in self.gravity:
            acc += model.compute_acc(pos)

        return acc

    # This method computes asteroid potential
    def compute_potential(self, pos):
        # Initialize potential
        U = 0

        # Loop through attached models
        for model in self.gravity:
            U += model.compute_U(pos)

        return U

    # This method set properties to asteroid class
    def set_properties(self, name,
                       mu=None, oe=None, eul313=None,
                       rot_period=None, file_shape=None):
        # Determine properties dictionary
        if name == 'eros':
            # Eros dictionary
            asteroid_dict = {'mu': 4.46275472004 * 1e5,
                             'oe': np.array([1.4583 * 149597870.7 * km2m,  # a
                                             0.2227,  # ecc
                                             10.829 * deg2rad,  # inc
                                             304.3 * deg2rad,  # RAAN
                                             178.9 * deg2rad,  # omega
                                             246.9 * deg2rad]),  # f
                             'eul313': [11.369 * deg2rad,  # ra
                                        17.227 * deg2rad,  # dec
                                        0 * deg2rad],  # lst0
                             'rot_period': 5.27 * 3600,
                             'file_shape': file_shape}
        elif name == 'custom':
            # Custom dictionary
            asteroid_dict = {'mu': mu,
                             'oe': np.array(oe),
                             'eul313': eul313,
                             'rot_period': rot_period,
                             'file_shape': file_shape}

        # Copy properties in class
        self.mu = asteroid_dict['mu']
        self.oe = asteroid_dict['oe']
        self.eul313 = asteroid_dict['eul313']
        self.rot_period = asteroid_dict['rot_period']

        # Set shape object
        self.shape = PolyhedronShape(asteroid_dict['file_shape'])

        # Compute heliocentric dcm at t=0
        self.dcm_N1N0 = rbk.euler3232C([self.eul313[0],
                                        np.pi/2 - self.eul313[1],
                                        self.eul313[2]])

    # This method deletes swigpy objects
    def delete_swigpy(self):
        self.shape.delete_shape()
        for model in self.gravity:
            model.delete_gravity()
