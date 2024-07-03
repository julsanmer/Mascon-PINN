import numpy as np


def full_initializer(shape, n_M):
    # Preallocate masses position
    xyz_M = np.zeros((n_M, 3))

    # Set limits for mascon masses
    xyz_surface = np.array(shape.xyzVertex)
    x_lim = np.array([np.min(xyz_surface[:, 0]),
                      np.max(xyz_surface[:, 0])])
    y_lim = np.array([np.min(xyz_surface[:, 1]),
                      np.max(xyz_surface[:, 1])])
    z_lim = np.array([np.min(xyz_surface[:, 2]),
                      np.max(xyz_surface[:, 2])])

    # Initialize masses counter
    cont = 0

    # Repeat after n_M masses are interior
    while cont < n_M:
        # Take random sample and compute its laplacian
        x = np.random.uniform(x_lim[0], x_lim[1])
        y = np.random.uniform(y_lim[0], y_lim[1])
        z = np.random.uniform(z_lim[0], z_lim[1])
        xyz = np.array([[x, y, z]])
        lap = shape.poly.computeLaplacian(xyz.tolist())

        # Check if it is an interior point
        if abs(lap) > 2*np.pi:
            xyz_M[cont, 0:3] = xyz
            cont += 1

    return xyz_M


def octant_initializer(shape, n_M):
    # Preallocate masses position
    xyz_M = np.zeros((n_M, 3))

    # Set limits for mascon masses
    xyz_surface = np.array(shape.xyz_vert)
    x_lim = np.array([np.min(xyz_surface[:, 0]),
                      np.max(xyz_surface[:, 0])])
    y_lim = np.array([np.min(xyz_surface[:, 1]),
                      np.max(xyz_surface[:, 1])])
    z_lim = np.array([np.min(xyz_surface[:, 2]),
                      np.max(xyz_surface[:, 2])])

    # Set number of masses per octant
    n_mod = n_M % 8
    if n_mod == 0:
        n_octant = np.ones(8) * int(n_M/8)
    else:
        rem = n_M - 8*int(n_M/8)
        n_octant = np.ones(8) * int(n_M/8) \
                   + np.concatenate((np.ones(rem),
                                     np.zeros(8-rem)))

    # Set octant limits and initialize counter
    x_octant = np.array([[x_lim[0], 0], [x_lim[0], 0],
                         [x_lim[0], 0], [x_lim[0], 0],
                         [0, x_lim[1]], [0, x_lim[1]],
                         [0, x_lim[1]], [0, x_lim[1]]])
    y_octant = np.array([[y_lim[0], 0], [y_lim[0], 0],
                         [0, y_lim[1]], [0, y_lim[1]],
                         [y_lim[0], 0], [y_lim[0], 0],
                         [0, y_lim[1]], [0, y_lim[1]]])
    z_octant = np.array([[z_lim[0], 0], [0, z_lim[1]],
                         [0, z_lim[1]], [z_lim[0], 0],
                         [z_lim[0], 0], [0, z_lim[1]],
                         [0, z_lim[1]], [z_lim[0], 0]])
    cont = 0

    # Loop through octants
    for i in range(8):
        # Initialize octant counter
        cont_octant = 0

        # Fill each octant
        while cont_octant < n_octant[i]:
            # Generate sample and compute its laplacian
            x = np.random.uniform(x_octant[i, 0], x_octant[i, 1])
            y = np.random.uniform(y_octant[i, 0], y_octant[i, 1])
            z = np.random.uniform(z_octant[i, 0], z_octant[i, 1])
            xyz = np.array([[x, y, z]]).squeeze()
            is_exterior = shape.check_exterior(xyz)

            # Check if it is an interior point
            if not is_exterior:
                xyz_M[cont, 0:3] = xyz
                cont_octant += 1
                cont += 1

    return xyz_M


def surface_initializer(shape, n_M):
    # Take random faces
    idx = rng.choice(len(xyz_face), size=n_M, replace=False)
    xyz_M = xyz_face[idx, 0:3]

    # Set mascon distribution output
    xyz_M = np.concatenate((np.array([[0, 0, 0]]),
                            xyz_M), axis=0)

    return xyz_M
