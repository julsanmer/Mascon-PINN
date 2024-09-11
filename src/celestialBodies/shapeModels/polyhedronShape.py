import numpy as np

from src.celestialBodies.shapeModels.shape import Shape

from Basilisk.fswAlgorithms import masconFit
from Basilisk.simulation import gravityEffector

km2m = 1e3


# Polyhedron
class PolyhedronShape(Shape):
    def __init__(self, file):
        super().__init__()

        # Set polyhedron file
        self.file = file

        # Load vertexes, face order,
        # number of vertexes and faces
        vert_list, face_list, n_vert, n_face = \
            gravityEffector.loadPolyFromFileToList(self.file)
        self.xyz_vert = np.array(vert_list)
        self.order_face = np.array(face_list)
        self.n_vert = n_vert
        self.n_face = n_face

        # Compute face centers and normals
        self.xyz_face = np.zeros((n_face, 3))
        self.normal_face = np.zeros((n_face, 3))
        for i in range(n_face):
            # Add face center
            idx = self.order_face[i, 0:3]
            self.xyz_face[i, 0:3] = (self.xyz_vert[idx[0]-1, 0:3]
                                     + self.xyz_vert[idx[1]-1, 0:3]
                                     + self.xyz_vert[idx[2]-1, 0:3]) / 3

            # Add face normal
            xyz_21 = self.xyz_vert[idx[1]-1, 0:3] \
                     - self.xyz_vert[idx[0]-1, 0:3]
            xyz_31 = self.xyz_vert[idx[2]-1, 0:3] \
                     - self.xyz_vert[idx[0]-1, 0:3]
            self.normal_face[i, 0:3] = np.cross(xyz_21, xyz_31) \
                                       / np.linalg.norm(np.cross(xyz_21, xyz_31))

        # Set axes
        a = (np.max(self.xyz_vert[:, 0])
             - np.min(self.xyz_vert[:, 0]))/2
        b = (np.max(self.xyz_vert[:, 1])
             - np.min(self.xyz_vert[:, 1]))/2
        c = (np.max(self.xyz_vert[:, 2])
             - np.min(self.xyz_vert[:, 2]))/2
        self.axes = np.array([a, b, c])

        # Set bsk shape
        self.shape_bsk = None
        self.create_shape()

    # This method creates a shape object
    def create_shape(self):
        # Create shape object
        masconfit_bsk = masconFit.MasconFit()
        self.shape_bsk = masconfit_bsk.shapeModel
        self.shape_bsk.xyzVertex = self.xyz_vert.tolist()
        self.shape_bsk.orderFacet = self.order_face.tolist()
        self.shape_bsk.initializeParameters()

    # This method checks if a point is exterior
    def check_exterior(self, pos):
        is_exterior = self.shape_bsk.isExterior(pos)

        return is_exterior

    # This method computes altitude
    def compute_altitude(self, pos):
        alt = self.shape_bsk.computeAltitude(pos)

        return alt

    # This method eliminates shape object
    # (important to save)
    def delete_shape(self):
        self.shape_bsk = None
