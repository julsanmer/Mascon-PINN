import numpy as np

from Basilisk.ExternalModules import gravEst


def initialize_mascon(gravest_bsk, mascon_init, seed_M, mu):
    # Extract vertexes and faces indexes
    xyz_vert = np.array(gravest_bsk.poly.xyzVertex)
    xyz_face = np.array(gravest_bsk.poly.xyzFacet)
    order_face = np.array(gravest_bsk.poly.orderFacet)

    # Set limits for mascon masses
    x_lim = np.array([np.min(xyz_vert[:, 0]), np.max(xyz_vert[:, 0])])
    y_lim = np.array([np.min(xyz_vert[:, 1]), np.max(xyz_vert[:, 1])])
    z_lim = np.array([np.min(xyz_vert[:, 2]), np.max(xyz_vert[:, 2])])

    # Retrieve number of masses
    n_M = gravest_bsk.mascon.nM - 1

    # Preallocate mascon distribution
    mu0_M = np.ones(n_M)
    pos0_M = np.zeros((n_M, 3))

    # Set initialization randomness
    np.random.seed(seed_M)

    # Choose type of initial distribution
    if mascon_init == 'full':
        # Initialize masses counter
        cont = 0

        # Repeat after n_M masses are interior
        while cont < n_M:
            # Take random sample and compute its laplacian
            x = np.random.uniform(x_lim[0], x_lim[1])
            y = np.random.uniform(y_lim[0], y_lim[1])
            z = np.random.uniform(z_lim[0], z_lim[1])
            pos = np.array([[x, y, z]])
            lap = gravest_bsk.poly.computeLaplacian(pos.tolist())

            # Check if it is an interior point
            if abs(lap[0][0]) > 2*np.pi:
                pos0_M[cont, 0:3] = pos
                cont += 1
    elif mascon_init == 'octant':
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
                lap = gravest_bsk.poly.computeLaplacian(pos.tolist())

                # Check if it is an interior point
                if abs(lap[0][0]) > 2*np.pi:
                    pos0_M[cont, 0:3] = pos
                    cont_octant += 1
                    cont += 1
    elif mascon_init == 'surface':
        # Take random faces
        idx = rng.choice(len(xyz_face), size=n_M, replace=False)
        pos0_M = xyz_face[idx, 0:3]

    # Prepare output
    pos0_M = np.concatenate((np.array([[0, 0, 0]]), pos0_M), axis=0)
    mu0_M = np.concatenate(([mu], mu0_M))

    return pos0_M, mu0_M


def train_mascon(parameters, outputs):
    # This function initializes bsk gravity estimation properties
    def initialize_gravest():
        # Set gravity estimation algorithm parameters
        gravest_bsk.maxIter = parameters.grav_est.maxiter
        gravest_bsk.lam = parameters.grav_est.lr
        gravest_bsk.useM = True
        gravest_bsk.useSH = False
        gravest_bsk.mascon.nM = int(parameters.grav_est.n_M) + 1

        # Set adimensional variables
        gravest_bsk.mascon.mu = parameters.grav_est.mu
        gravest_bsk.mascon.muMad = parameters.grav_est.muM_ad
        gravest_bsk.mascon.posMad = parameters.grav_est.posM_ad.tolist()

        # Choose algorithm and loss function
        gravest_bsk.useAdam = True
        gravest_bsk.useAdagrad = False
        gravest_bsk.useNAGD = False
        gravest_bsk.useMSE = True
        gravest_bsk.useMLE = False

        # Set training variables flag
        parameters.grav_est.mascon_type = 'MUPOS'
        if parameters.grav_est.mascon_type == 'MU':
            gravest_bsk.mascon.MU = True
            gravest_bsk.mascon.MUPOS = False
        elif parameters.grav_est.mascon_type == 'MUPOS':
            gravest_bsk.mascon.MU = False
            gravest_bsk.mascon.MUPOS = True

        # Set polyhedron
        gravest_bsk.poly.nVertex = parameters.grav_est.n_vert
        gravest_bsk.poly.nFacet = parameters.grav_est.n_face
        gravest_bsk.poly.xyzVertex = parameters.grav_est.xyz_vert.tolist()
        gravest_bsk.poly.orderFacet = parameters.grav_est.order_face.tolist()
        gravest_bsk.poly.initializeParameters()

        # Initialize mascon distribution
        pos0_M, mu0_M = initialize_mascon(gravest_bsk, parameters.grav_est.mascon_init,
                                          parameters.grav_est.seed_M, parameters.grav_est.mu)
        parameters.grav_est.pos0_M = pos0_M
        parameters.grav_est.mu0_M = mu0_M
        gravest_bsk.mascon.posM = pos0_M.tolist()
        gravest_bsk.mascon.muM = mu0_M.tolist()

    # Initialize gravity estimation module
    gravest_bsk = gravEst.GravEst()
    initialize_gravest()

    # Do training loop
    n_segments = len(parameters.times_groundtruth) - 1
    t_data = outputs.groundtruth.t
    r_data = outputs.groundtruth.r_CA_A
    a_data = outputs.groundtruth.a_A
    mu = parameters.asteroid.mu

    ## Perturb tData to just account for eclipse
    #if data.dataType == 'orbit' and mascon.eclipseFlag:
    #    tData += (data.navFlag-1)*1.5*trainPeriods[-1,1]

    ## Reset random generator
    #rng = np.random.default_rng(mascon.randSeed)

    # Loop through data batches
    for i in range(n_segments):
        # Print status
        print(str(i) + '/' + str(n_segments) + ' segments completed' + '; ' + parameters.grav_est.mascon_type
              + ' n=' + str(int(parameters.grav_est.n_M)))
        # Select data segment
        t0 = parameters.times_dmcukf[i, 0]
        tf = parameters.times_dmcukf[i, 1]
        idx = np.where(np.logical_and(t_data >= t0, t_data < tf))[0]

        # Extract data segment
        r_batch = r_data[idx, 0:3]
        a_batch = a_data[idx, 0:3]
        #if mascon.ejectaFlag:
        #    idxEjecta = rng.choice(len(posData_ii), size=mascon.Nejectaorbit, replace=False)
        #    posData_ii[idxEjecta,0:3] = data.posEjecta[0,0:mascon.Nejectaorbit,0:3]
        #    accData_ii[idxEjecta,0:3] = data.accEjecta[0,0:mascon.Nejectaorbit,0:3]

        for j in range(len(r_batch)):
            a_batch[j, 0:3] += -mu*r_batch[j, 0:3] / np.linalg.norm(r_batch[j, 0:3])**3

        # Train data
        W = np.ones(3*len(r_batch))
        gravest_bsk.trainGravity(r_batch.tolist(), a_batch.tolist(), W.tolist())

    # Extract trained mascons
    parameters.grav_est.mu_M = np.array(gravest_bsk.mascon.muM)
    parameters.grav_est.pos_M = np.array(gravest_bsk.mascon.posM)
        #L = np.array(gravest_bsk.L)
        ##plt.plot(J, marker='.')
        ##plt.show()
        #if i == 0:
        #    iter = np.zeros((len(L), len(data.trainPeriods)))
        #    loss = np.zeros((len(L), len(data.trainPeriods)))
        #iter[:,i] = (np.linspace(1,len(L),len(L))+i*len(L))/len(L)
        #loss[:,i] = L.squeeze()
        ##if ii == 0:
        ##    break
