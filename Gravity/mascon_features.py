import numpy as np

from Basilisk.ExternalModules import masconFit


def initialize_mascon(masconfit_bsk, grav_est):
    # Extract vertexes and faces indexes
    xyz_vert = np.array(grav_est.xyz_vert)
    xyz_face = np.array(grav_est.xyz_face)
    order_face = np.array(grav_est.order_face)

    # Set limits for mascon masses
    x_lim = np.array([np.min(xyz_vert[:, 0]), np.max(xyz_vert[:, 0])])
    y_lim = np.array([np.min(xyz_vert[:, 1]), np.max(xyz_vert[:, 1])])
    z_lim = np.array([np.min(xyz_vert[:, 2]), np.max(xyz_vert[:, 2])])

    # Retrieve number of masses
    n_M = masconfit_bsk.nM - 1

    # Preallocate mascon distribution
    mu0_M = np.ones(n_M)
    pos0_M = np.zeros((n_M, 3))

    # Set initialization randomness
    np.random.seed(grav_est.seed_M)

    # Choose type of initial distribution
    if grav_est.mascon_init == 'full':
        # Initialize masses counter
        cont = 0

        # Repeat after n_M masses are interior
        while cont < n_M:
            # Take random sample and compute its laplacian
            x = np.random.uniform(x_lim[0], x_lim[1])
            y = np.random.uniform(y_lim[0], y_lim[1])
            z = np.random.uniform(z_lim[0], z_lim[1])
            pos = np.array([[x, y, z]])
            lap = masconfit_bsk.poly.computeLaplacian(pos.tolist())

            # Check if it is an interior point
            if abs(lap[0][0]) > 2*np.pi:
                pos0_M[cont, 0:3] = pos
                cont += 1
    elif grav_est.mascon_init == 'octant':
        # Set number of masses per octant
        n_mod = n_M % 8
        if n_mod == 0:
            n_octant = np.ones(8) * int(n_M / 8)
        else:
            rem = n_M - 8 * int(n_M / 8)
            n_octant = np.ones(8) * int(n_M / 8) + np.concatenate((np.ones(rem), np.zeros(8-rem)))

        # Set octant limits and initialize counter
        x_octant = np.array([[x_lim[0], 0], [x_lim[0], 0], [x_lim[0], 0], [x_lim[0], 0],
                            [0, x_lim[1]], [0, x_lim[1]], [0, x_lim[1]], [0, x_lim[1]]])
        y_octant = np.array([[y_lim[0], 0], [y_lim[0], 0], [0, y_lim[1]], [0, y_lim[1]],
                            [y_lim[0], 0], [y_lim[0], 0], [0, y_lim[1]], [0, y_lim[1]]])
        z_octant = np.array([[z_lim[0], 0], [0, z_lim[1]], [0, z_lim[1]], [z_lim[0], 0],
                            [z_lim[0], 0], [0, z_lim[1]], [0, z_lim[1]], [z_lim[0], 0]])
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
                pos = np.array([[x, y, z]])
                lap = masconfit_bsk.shape.computeLaplacianBatch(pos.tolist())

                # Check if it is an interior point
                if abs(lap[0][0]) > 2*np.pi:
                    pos0_M[cont, 0:3] = pos
                    cont_octant += 1
                    cont += 1
    elif grav_est.mascon_init == 'surface':
        # Take random faces
        idx = rng.choice(len(xyz_face), size=n_M, replace=False)
        pos0_M = xyz_face[idx, 0:3]

    # Prepare output
    pos0_M = np.concatenate((np.array([[0, 0, 0]]), pos0_M), axis=0)
    mu0_M = np.concatenate(([grav_est.mu], mu0_M))

    return pos0_M, mu0_M


def train_mascon(parameters, outputs):
    # This function initializes bsk gravity estimation properties
    def initialize_gravest():
        # Set gravity estimation algorithm parameters
        masconfit_bsk.nM = int(parameters.grav_est.n_M) + 1

        # Set adimensional variables
        masconfit_bsk.mu = parameters.grav_est.mu
        masconfit_bsk.muMad = parameters.grav_est.muM_ad
        masconfit_bsk.posMad = parameters.grav_est.posM_ad.tolist()

        # Choose algorithm and loss function
        masconfit_bsk.useMSE = True
        masconfit_bsk.useMLE = False

        # Set training variables flag
        if parameters.grav_est.mascon_type == 'MU':
            masconfit_bsk.trainPOS = False
        elif parameters.grav_est.mascon_type == 'MUPOS':
            masconfit_bsk.trainPOS = True

        # Set polyhedron
        masconfit_bsk.shape.initPolyhedron(parameters.grav_est.xyz_vert.tolist(),
                                           parameters.grav_est.order_face.tolist())

        # Initialize mascon distribution
        pos0_M, mu0_M = initialize_mascon(masconfit_bsk, parameters.grav_est)
        parameters.grav_est.pos0_M = pos0_M
        parameters.grav_est.mu0_M = mu0_M
        masconfit_bsk.posM = pos0_M.tolist()
        masconfit_bsk.muM = mu0_M.tolist()

    # Initialize gravity estimation module
    masconfit_bsk = masconFit.MasconFit()
    initialize_gravest()

    # Do training loop
    n_segments = len(parameters.times_groundtruth) - 1
    t_data = outputs.groundtruth.t
    pos_data = outputs.groundtruth.pos_BP_P
    acc_data = outputs.groundtruth.acc_BP_P
    mu = parameters.asteroid.mu

    # Loop through data batches
    for i in range(n_segments):
        # Select data segment
        t0 = parameters.times_dmcukf[i, 0]
        tf = parameters.times_dmcukf[i, 1]
        idx = np.where(np.logical_and(t_data >= t0, t_data < tf))[0]

        # Extract data segment
        pos_batch = pos_data[idx, 0:3]
        acc_batch = acc_data[idx, 0:3]
        #if mascon.ejectaFlag:
        #    idxEjecta = rng.choice(len(posData_ii), size=mascon.Nejectaorbit, replace=False)
        #    posData_ii[idxEjecta,0:3] = data.posEjecta[0,0:mascon.Nejectaorbit,0:3]
        #    accData_ii[idxEjecta,0:3] = data.accEjecta[0,0:mascon.Nejectaorbit,0:3]

        # Train data
        masconfit_bsk.train(pos_batch.tolist(), acc_batch.tolist())

    # Extract trained mascons
    parameters.grav_est.mu_M = np.array(masconfit_bsk.muM)
    parameters.grav_est.pos_M = np.array(masconfit_bsk.posM)
