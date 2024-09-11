import numpy as np


# This class defines a surface map
class MapSurface3D:
    def __init__(self):
        # Preallocate altitude over terrain
        self.alt = 1e-3

        # Preallocate evaluation and acceleration
        self.xyz_surf = []
        self.acc_surf = []
        self.aErr_surf = []

    # This method creates surface grid
    def create_grid(self, shape):
        # Take center of facets
        self.xyz_surf = shape.xyz_face \
                        + self.alt * shape.normal_face

    # This method computes surface acceleration
    def generate_acc(self, grav_model):
        # Preallocate variables
        n = len(self.xyz_surf)
        self.acc_surf = np.zeros((n, 3))

        # Loop through surface
        for i in range(n):
            self.acc_surf[i, 0:3] =\
                grav_model.compute_gravity(self.xyz_surf[i, 0:3])

    # This computes surface error map
    def compute_errors(self, refmap_surf):
        # Compute 3D gravity map error
        self.aErr_surf = np.linalg.norm(self.acc_surf - refmap_surf.acc_surf, axis=1) \
                         / np.linalg.norm(refmap_surf.acc_surf, axis=1)

    # This method imports surface map
    def import_grid(self, refmap_surf):
        # Copy surface map evaluation
        self.alt = refmap_surf.alt
        self.xyz_surf = refmap_surf.xyz_surf
