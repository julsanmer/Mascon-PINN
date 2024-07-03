import numpy as np

from Basilisk.fswAlgorithms import masconFit


# This function initializes the mascon distribution
def check_collision(asteroid, pos_BP_P):
    # This function initializes BSK macon fit module
    def initialize_shape():
        # Set polyhedron
        shape.xyzVertex = asteroid.xyz_vert.tolist()
        shape.orderFacet = asteroid.order_face.tolist()
        shape.initializeParameters()

    def check_exterior(pos):
        # Initialize flag
        outCuboid = False

        # Check if it lies outside cuboid
        if pos[0] > x_lim[1] or pos[0] < x_lim[0]:
            outCuboid = True
            if pos[1] > y_lim[1] or pos[1] < y_lim[0]:
                outCuboid = True
                if pos[2] > z_lim[1] or pos[2] < z_lim[0]:
                    outCuboid = True

        # Check if it lies outside cuboid
        if outCuboid:
            isExterior = True
        else:
            isExterior = shape.isExterior(pos)

        return isExterior

    # Initialize gravity estimation module
    masconfit_bsk = masconFit.MasconFit()
    shape = masconfit_bsk.shapeModel
    initialize_shape()

    # Set cuboid limits
    x_lim = np.array([np.min(asteroid.xyz_vert[:, 0]),
                      np.max(asteroid.xyz_vert[:, 0])])
    y_lim = np.array([np.min(asteroid.xyz_vert[:, 1]),
                      np.max(asteroid.xyz_vert[:, 1])])
    z_lim = np.array([np.min(asteroid.xyz_vert[:, 2]),
                      np.max(asteroid.xyz_vert[:, 2])])

    # Loop through positions
    n = len(pos_BP_P)
    for i in range(n):
        # Check exterior
        isExterior = check_exterior(pos_BP_P[i, 0:3])

        # If there is a collision
        if not isExterior:
            # Set all future positions to nan
            pos_BP_P[i:n, 0:3] = pos_BP_P[i:n, 0:3]*np.nan

            break

    return pos_BP_P
