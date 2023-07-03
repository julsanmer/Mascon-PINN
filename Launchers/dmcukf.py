import numpy as np

from Basilisk.ExternalModules import smallbodyDMCUKF
from Basilisk.ExternalModules import gravEst
from Basilisk.utilities import unitTestSupport as sp
from Basilisk.utilities import RigidBodyKinematics

from Gravity.mascon_features import initialize_mascon


def run_dmcukf(parameters, outputs):
    # This function initializes bsk dmc-ukf properties
    def initialize_dmcukf():
        # Set dmc-ukf mascon model
        dmcukf_bsk.useSH = False
        dmcukf_bsk.useM = True
        dmcukf_bsk.mascon.nM = int(parameters.grav_est.n_M + 1)
        dmcukf_bsk.mascon.posM = parameters.grav_est.pos0_M.tolist()
        dmcukf_bsk.mascon.muM = parameters.grav_est.mu0_M.tolist()

        # Set solar radiation pressure if required
        if parameters.flag_sun:
            dmcukf_bsk.useSRP = True
            dmcukf_bsk.useSun = True
            dmcukf_bsk.m = parameters.spacecraft.mass
            dmcukf_bsk.ASRP = parameters.spacecraft.srpArea
            dmcukf_bsk.CR = parameters.spacecraft.CR

        # Set dmc-ukf state, uncertainty, process and measurement covariances
        dmcukf_bsk.xhat_k = sp.np2EigenVectorXd(parameters.dmcukf.x_k + parameters.dmcukf.dx_k)
        dmcukf_bsk.Pxx_k = parameters.dmcukf.P_k.tolist()
        dmcukf_bsk.Pww = parameters.dmcukf.P_proc.tolist()
        dmcukf_bsk.Pvv = parameters.dmcukf.R_meas.tolist()

        # Set hyperparameters
        dmcukf_bsk.alpha = parameters.dmcukf.alpha
        dmcukf_bsk.beta = parameters.dmcukf.beta
        dmcukf_bsk.kappa = parameters.dmcukf.kappa
        dmcukf_bsk.Nint = parameters.dmcukf.n_prop

        # Set what type of measurements is used
        dmcukf_bsk.useMeasSimple = False
        dmcukf_bsk.useMeasPoscam = False
        dmcukf_bsk.useMeasPixel = True

        # Set landmarks information
        dmcukf_bsk.nLandmarks = len(parameters.sensors.xyz_lmk)
        dmcukf_bsk.xyzLandmarks = parameters.sensors.xyz_lmk + parameters.sensors.dxyz_lmk
        dmcukf_bsk.f = parameters.sensors.f
        dmcukf_bsk.wPixel = parameters.sensors.w_pixel

        # Set initial small body lst0 and rotational period
        dmcukf_bsk.lst0 = parameters.asteroid.lst0
        dmcukf_bsk.rotRate = 2*np.pi / parameters.asteroid.rot_period

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

    # This function loads measurements into bsk dmc-ukf module
    def load_measurements():
        # Extract corresponding batch of measurements
        idx = np.where(np.logical_and(t >= t0, t <= tf))[0]
        if i > 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        t_meas = t[idx].tolist()

        # Extract small body position and orientation
        posTruth_AS_N = outputs.groundtruth.pos_AS_N[idx, :].tolist()
        eul323Truth_AN = outputs.groundtruth.eul323_AN[idx, :].tolist()

        # Extract camera variables
        pixel = outputs.camera.pixel[idx, :].tolist()
        n_visible = outputs.camera.n_visible[idx].tolist()
        visible = outputs.camera.flag_nav[idx].tolist()
        latlon = outputs.camera.latlon[idx, :].tolist()

        # Load measurements into module
        dmcukf_bsk.statemeasBatch.tBatch = t_meas
        dmcukf_bsk.statemeasBatch.nvisibleBatch = n_visible
        dmcukf_bsk.statemeasBatch.visibleBatch = visible
        dmcukf_bsk.statemeasBatch.pixelBatch = pixel
        dmcukf_bsk.statemeasBatch.latlonBatch = latlon
        dmcukf_bsk.statemeasBatch.rBatch_AS = posTruth_AS_N
        dmcukf_bsk.statemeasBatch.eul323Batch_AN = eul323Truth_AN
        dmcukf_bsk.statemeasBatch.nSegment = i

        return t_meas

    # This function saves dmc-ukf results
    def save_dmcukf():
        tcpu_dmcukf = np.array(dmcukf_bsk.statemeasBatch.tcpu)

        # Obtain filter state, covariance, training data and residuals
        Xhat = np.array(dmcukf_bsk.statemeasBatch.Xhat)
        Pxx_array = np.array(dmcukf_bsk.statemeasBatch.Pxx)
        # resZ = np.array(DMCUKF.statebatch.resZ)
        r_data = np.array(dmcukf_bsk.statemeasBatch.rGrav)
        a_data = np.array(dmcukf_bsk.statemeasBatch.aGrav)

        # Set estimated variables
        n = len(Xhat)
        rEst_CA_N = Xhat[:, 0:3]
        vEst_CA_N = Xhat[:, 3:6]
        aEst_N = Xhat[:, 6:9]
        P = Pxx_array.reshape((len(t_meas), 9, 9))

        # Initialize variables in small body frame
        rEst_CA_A = np.zeros((n, 3))
        Ppos_A = np.zeros((n, 3, 3))
        Pacc_A = np.zeros((n, 3, 3))

        # Fill variables in small body frame
        for k in range(n):
            lst = parameters.asteroid.lst0 + 2 * np.pi / parameters.asteroid.rot_period * t_meas[k]
            eul323 = np.array([0, 0, lst])
            dcm_AN = RigidBodyKinematics.euler3232C(eul323)
            rEst_CA_A[k, 0:3] = np.matmul(dcm_AN, rEst_CA_N[k, 0:3])
            Ppos_A[k, 0:3, 0:3] = np.matmul(dcm_AN, np.matmul(P[k, 0:3, 0:3], dcm_AN.transpose()))
            Pacc_A[k, 0:3, 0:3] = np.matmul(dcm_AN, np.matmul(P[k, 6:9, 6:9], dcm_AN.transpose()))

        # Save variables in outputs
        if i == 0:
            # Initialize position and velocity
            outputs.results.pos_CA_N = rEst_CA_N
            outputs.results.vel_CA_N = vEst_CA_N

            # Save acceleration
            outputs.results.acc_CA_N = aEst_N

            # Save training variables
            outputs.results.r_data = r_data
            outputs.results.a_data = a_data

            # Save covariance and variables in small body frame
            outputs.results.P = P
            outputs.results.pos_CA_A = rEst_CA_A
            outputs.results.Ppos_A = Ppos_A
            outputs.results.Pacc_A = Pacc_A

            # Save filter times
            outputs.results.tcpu_dmcukf = tcpu_dmcukf[1:n]

            ## Save residual
            # navOutputs.nav.resZ = resZ
        else:
            # Fill variables
            outputs.results.pos_CA_N = np.concatenate((outputs.results.pos_CA_N, rEst_CA_N[1:n, :]))
            outputs.results.vel_CA_N = np.concatenate((outputs.results.vel_CA_N, vEst_CA_N[1:n, :]))
            outputs.results.acc_CA_N = np.concatenate((outputs.results.acc_CA_N, aEst_N[1:n, :]))
            outputs.results.r_data = np.concatenate((outputs.results.r_data, rEst_CA_A[1:n, :]))
            outputs.results.a_data = np.concatenate((outputs.results.a_data, a_data[1:n, :]))
            outputs.results.P = np.concatenate((outputs.results.P, P[1:n, :, :]))
            outputs.results.pos_CA_A = np.concatenate((outputs.results.pos_CA_A, rEst_CA_A[1:n, :]))
            outputs.results.Ppos_A = np.concatenate((outputs.results.Ppos_A, Ppos_A[1:n, :, :]))
            outputs.results.Pacc_A = np.concatenate((outputs.results.Pacc_A, Pacc_A[1:n, :, :]))

            # Save dmc-ukf computational times
            outputs.results.tcpu_dmcukf = np.concatenate((outputs.results.tcpu_dmcukf,
                                                          tcpu_dmcukf[1:n]))

    # This function prepares data for bsk gravity estimation
    def prepare_gravest():
        # Extract indexes
        idx = np.where(np.logical_and(t >= parameters.times_dmcukf[i, 0],
                                      t < parameters.times_dmcukf[i, 1]))[0]
        flag_nav = outputs.camera.flag_nav[idx]
        idx = idx[flag_nav == 1]

        # Preallocate data batch and weights
        t_data = t[idx]
        r_data = outputs.results.r_data[idx]
        a_data = outputs.results.a_data[idx]
        Wvec = np.ones(3 * int(np.sum(flag_nav)))

        # # Filter posBatch, accBatch
        # intBatch = parameters.grav_rate / parameters.dmcukf_rate
        # lenBatch = len(tBatch) - 1
        # idxBatch = np.linspace(0, np.floor(lenBatch / intBatch) * intBatch,
        #                        int(np.floor(lenBatch / intBatch)) + 1).astype(int)
        # tBatch = tBatch[idxBatch]
        # posBatch = posBatch[idxBatch, 0:3]
        # accBatch = accBatch[idxBatch, 0:3]

        # Include ejecta if needed
        if parameters.grav_est.flag_ejecta:
            idx = rng.choice(len(r_data), size=parameters.grav_est.n_ejecta, replace=False)
            r_data[idx, 0:3] = outputs.groundtruth.pos_EA_A[0, 0:parameters.grav_est.n_ejecta, 0:3]
            a_data[idx, 0:3] = outputs.groundtruth.acc_EA_A[0, 0:parameters.grav_est.n_ejecta, 0:3]

        return t_data, r_data, a_data, Wvec

    # ------------------- SIMULTANEOUS NAVIGATION AND GRAVITY ESTIMATION ------------------------- #
    # Set rng
    rng = np.random.default_rng(0)

    # Retrieve ground truth times
    t = outputs.groundtruth.t
    n_segments = len(parameters.times_groundtruth) - 1

    # Set dmc-ukf initial condition
    pos0 = outputs.groundtruth.pos_CA_N[0, :]
    vel0 = outputs.groundtruth.vel_CA_N[0, :]
    acc0 = np.zeros(3)
    parameters.dmcukf.x_k = np.concatenate((pos0, vel0, acc0))

    # Preallocate mascon distribution
    outputs.results.mu_M = np.zeros((parameters.grav_est.n_M+1, n_segments+1))
    outputs.results.pos_M = np.zeros((parameters.grav_est.n_M+1, 3, n_segments+1))

    # Preallocate computational times and loss
    outputs.results.tcpu_gravest = np.zeros(n_segments)
    outputs.results.loss = np.zeros((parameters.grav_est.maxiter, n_segments))

    # Initialize basilisk gravity estimation module
    gravest_bsk = gravEst.GravEst()
    initialize_gravest()

    # Initialize basilisk dmc-ukf module
    dmcukf_bsk = smallbodyDMCUKF.SmallbodyDMCUKF()
    initialize_dmcukf()
    dmcukf_bsk.Reset(0)

    # Loop through simulation segments
    for i in range(n_segments):
        # Print status
        print(str(i) + '/' + str(n_segments) + ' segments completed')

        # Load measurements batch
        t0 = parameters.times_groundtruth[i]
        tf = parameters.times_groundtruth[i+1]
        t_meas = load_measurements()

        # Reset acceleration to zero after fit
        if i > 0:
            xhat_k = np.array(dmcukf_bsk.xhat_k)
            xhat_k[6:9] = 0
            dmcukf_bsk.xhat_k = xhat_k.tolist()

        # Run filter forward
        dmcukf_bsk.updateMeasBatch()

        # Save data
        save_dmcukf()

        # Prepare data for gravity estimation
        t_data, r_data, a_data, Wvec = prepare_gravest()

        # Do gravity fit
        gravest_bsk.trainGravity(r_data.tolist(), a_data.tolist(), Wvec.tolist())
        outputs.results.loss[:, i] = np.array(gravest_bsk.L).squeeze()
        outputs.results.tcpu_gravest[i] = gravest_bsk.tcpu

        # Set gravity variables into dmc-ukf
        dmcukf_bsk.mascon.posM = gravest_bsk.mascon.posM
        dmcukf_bsk.mascon.muM = gravest_bsk.mascon.muM

        # Fill output variables
        outputs.results.mu_M[:, i+1] = np.array(gravest_bsk.mascon.muM).squeeze()
        outputs.results.pos_M[:, 0:3, i+1] = np.array(gravest_bsk.mascon.posM).squeeze()

        # Reset some mascon to the unity
        muMtemp = np.array(gravest_bsk.mascon.muM)
        muMtemp[muMtemp < 1] = 1
        gravest_bsk.mascon.muM = muMtemp.tolist()

        # # Save trained variables
        # nTrain = len(t_data)
        # outputs.results.t_data = t_data
        # outputs.results.a_data = a_data

    parameters.grav_est.pos_M = np.array(dmcukf_bsk.mascon.posM)
    parameters.grav_est.mu_M = np.array(dmcukf_bsk.mascon.muM)
