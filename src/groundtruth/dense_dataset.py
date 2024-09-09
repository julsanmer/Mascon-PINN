import numpy as np


# This function generates a dataset
# around the asteroid
def generate_dense_dataset(sc, asteroid, n_data,
                           rmax=None, type='alt'):
    # This draws a spherical sample
    def rad_sample(rmax):
        # Create a random spherical sample
        lat = np.random.uniform(-np.pi / 2, np.pi / 2)
        lon = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, rmax)
        pos_sample = r * np.array([np.cos(lat) * np.cos(lon),
                                   np.cos(lat) * np.sin(lon),
                                   np.sin(lat)])

        return pos_sample

    # This draws an ellipsoid sample
    def ell_sample(rmax, axes):
        # Do a random position sample
        a, b, c = axes
        lat = np.random.uniform(-np.pi / 2, np.pi / 2)
        lon = np.random.uniform(0, 2 * np.pi)
        s = np.random.uniform(0, rmax / np.min([a, b, c]))
        pos_sample = s * np.array([a * np.cos(lat) * np.cos(lon),
                                   b * np.cos(lat) * np.sin(lon),
                                   c * np.sin(lat)])

        return pos_sample

    # This draws a shape-based sample
    def alt_sample(hmax, xyz_face):
        # Get faces
        n_face = len(xyz_face)

        # Do a random position sample
        idx = np.random.choice(n_face)
        h = np.random.uniform(0, hmax)
        pos_surface = xyz_face[idx, 0:3]
        r_surface = np.linalg.norm(pos_surface)
        pos_sample = pos_surface * (1 + h/r_surface)

        return pos_sample

    # Extract shape instance
    shape = asteroid.shape

    # Preallocate dataset
    n = n_data
    pos_BP_P = np.zeros((n, 3))
    acc_BP_P = np.zeros((n, 3))
    r_BP, h_BP = np.zeros(n), np.zeros(n)
    U = np.zeros(n)

    # Set type of dense data
    if type == 'alt':
        # Set maximum altitude
        hmax = rmax
        n_face = shape.n_face
        xyz_face = shape.xyz_face

    # Create position-gravity acceleration time
    print('------- Initiating ground truth -------')
    np.random.seed(0)
    for i in range(n):
        # Set exterior point flag to false
        flag_ext = False
        while not flag_ext:
            if type == 'rad':
                pos = rad_sample(rmax)
            elif type == 'ell':
                pos = ell_sample(rmax, axes)
            elif type == 'alt':
                pos = alt_sample(hmax, xyz_face)

            # Check if point is exterior
            is_exterior = shape.check_exterior(pos.tolist())

            # Evaluate only if it is an exterior point
            if is_exterior:
                # Compute gravity and potential
                pos_BP_P[i, 0:3] = pos
                acc_BP_P[i, 0:3] = asteroid.compute_gravity(pos)
                U[i] = asteroid.compute_potential(pos)

                # Compute radius and altitude
                r_BP[i] = np.linalg.norm(pos)
                h_BP[i] = asteroid.shape.compute_altitude(pos)

                # Set flag to exit
                flag_ext = True

        # Print status
        if (i + 1) % int(n / 100) == 0:
            print(str(int((i + 1) / n * 100)) + ' % data generated')

    # Output status
    print('------- Finished ground truth -------')

    # Save generated dataset
    data = sc.data
    data.pos_BP_P = pos_BP_P
    data.acc_BP_P = acc_BP_P
    data.U = U
    data.r_BP = r_BP
    data.h_BP = h_BP
