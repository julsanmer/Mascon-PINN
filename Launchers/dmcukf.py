import numpy as np

from Basilisk.utilities import unitTestSupport as sp
from Basilisk.utilities import RigidBodyKinematics as rbk

from Basilisk.ExternalModules import dmcukf
from Basilisk.ExternalModules import masconFit

from Gravity.mascon_features import initialize_mascon
from Gravity.gravity_map import gravity_series


# This function executes the simultaneous navigation
# and gravity estimation algorithms
def run_dmcukf(parameters, outputs):
    # This function initializes bsk dmc-ukf properties
    def initialize_dmcukf():
        # Set dmc-ukf mascon model
        dmcukf_bsk.setMascon(parameters.grav_est.mu0_M.tolist(),
                             parameters.grav_est.pos0_M.tolist())

        # Set SRP and Sun's 3rd body if required
        if parameters.flag_sun:
            dmcukf_bsk.useSRP = True
            dmcukf_bsk.useSun = True
            dmcukf_bsk.m = parameters.spacecraft.mass
            dmcukf_bsk.ASRP = parameters.spacecraft.srpArea
            dmcukf_bsk.CR = parameters.spacecraft.CR

        # Set dmc-ukf state, uncertainty, process and measurement covariances
        dmcukf_bsk.x = sp.np2EigenVectorXd(parameters.dmcukf.x_k
                                           + parameters.dmcukf.dx_k)
        dmcukf_bsk.Pxx = parameters.dmcukf.P_k.tolist()
        dmcukf_bsk.Pproc = parameters.dmcukf.P_proc.tolist()
        dmcukf_bsk.Ppixel = parameters.dmcukf.P_pixel.tolist()

        # Set integration steps
        dmcukf_bsk.Nint = parameters.dmcukf.n_prop

        # Set landmark and camera information
        dmcukf_bsk.nLmk = len(parameters.sensors.xyz_lmk)
        dmcukf_bsk.rLmk = parameters.sensors.xyz_lmk \
                          + parameters.sensors.dxyz_lmk
        dmcukf_bsk.f = parameters.sensors.f
        dmcukf_bsk.wPixel = parameters.sensors.w_pixel

        # Set dcm of planet equatorial w.r.t. inertial and
        # camera w.r.t. body frame
        dmcukf_bsk.dcm_N1N0 = parameters.asteroid.dcm_N1N0
        dmcukf_bsk.dcm_CB = parameters.sensors.dcm_CB

    # This function initializes bsk gravity estimation properties
    def initialize_masconfit():
        # Set gravity estimation algorithm parameters
        masconfit_bsk.nM = int(parameters.grav_est.n_M) + 1

        # Set adimensional variables
        masconfit_bsk.mu = parameters.grav_est.mu
        masconfit_bsk.muMad = parameters.grav_est.muM_ad
        masconfit_bsk.posMad = parameters.grav_est.posM_ad.tolist()

        # Choose loss function type
        masconfit_bsk.useMSE = True
        masconfit_bsk.useMLE = False

        # Set training variables flag
        if parameters.grav_est.mascon_type == 'MUPOS':
            masconfit_bsk.trainPOS = True
        else:
            masconfit_bsk.trainPOS = False

        # Set polyhedron shape
        masconfit_bsk.shape.initPolyhedron(parameters.grav_est.xyz_vert.tolist(),
                                           parameters.grav_est.order_face.tolist())

        # Initialize mascon distribution
        pos0_M, mu0_M = initialize_mascon(masconfit_bsk, parameters.grav_est)
        parameters.grav_est.pos0_M = pos0_M
        parameters.grav_est.mu0_M = mu0_M
        masconfit_bsk.posM = pos0_M.tolist()
        masconfit_bsk.muM = mu0_M.tolist()

    # This function loads measurements into bsk dmc-ukf module
    def load_measurements():
        # Extract corresponding batch of measurements
        idx = np.where(np.logical_and(t >= t0, t <= tf))[0]
        if i > 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        t_meas = t[idx].tolist()

        # Extract small body position and orientation
        pos_PS_N1 = outputs.groundtruth.pos_PS_N1[idx, :].tolist()
        mrp_PN0 = outputs.groundtruth.mrp_PN0[idx, :].tolist()
        mrp_BP = outputs.groundtruth.mrp_BP[idx, :].tolist()

        # Load small body ephemeris and orientation
        dmcukf_bsk.batch.r_PS_N1 = pos_PS_N1
        dmcukf_bsk.batch.mrp_PN0 = mrp_PN0

        # Loop spacecraft orientation w.r.t. small body rotating frame
        dmcukf_bsk.batch.mrp_BP = mrp_BP

        # Extract camera variables
        pixel_lmk = outputs.camera.pixel_lmk[idx, :].tolist()
        isvisible_lmk = outputs.camera.isvisible_lmk[idx, :].tolist()
        mrp_CP = outputs.camera.mrp_CP[idx, :].tolist()

        # Load measurements into module
        dmcukf_bsk.batch.t = t_meas
        dmcukf_bsk.batch.isvisibleLmk = isvisible_lmk
        dmcukf_bsk.batch.pLmk = pixel_lmk

    # This function saves dmc-ukf results
    def save_dmcukf():
        # Obtain filter state, covariance, training data and residuals
        X = np.array(dmcukf_bsk.batch.x)
        Pxx_array = np.array(dmcukf_bsk.batch.Pxx)

        # Get small body orientation variables
        dcm_N1N0 = np.array(dmcukf_bsk.dcm_N1N0)
        mrp_PN0 = np.array(dmcukf_bsk.batch.mrp_PN0)

        # Set estimated variables
        n = len(X)
        pos_BP_N1 = X[:, 0:3]
        vel_BP_N1 = X[:, 3:6]
        acc_BP_N1 = X[:, 6:9]
        Pxx = Pxx_array.reshape((n, 9, 9))

        # Initialize variables in small body frame
        pos_data = np.zeros((n, 3))
        acc_data = np.zeros((n, 3))
        Ppos_P = np.zeros((n, 3, 3))
        Pacc_P = np.zeros((n, 3, 3))

        # Fill variables in small body frame
        for k in range(n):
            # Obtain spacecraft to planet dcm
            dcm_PN0 = rbk.MRP2C(mrp_PN0[k, 0:3])
            dcm_PN1 = np.matmul(dcm_PN0, dcm_N1N0.T)

            # Compute data position and acceleration
            pos_data[k, 0:3] = dcm_PN1.dot(pos_BP_N1[k, 0:3])
            acc_data[k, 0:3] = dcm_PN1.dot(acc_BP_N1[k, 0:3]) \
                               + np.array(dmcukf_bsk.mascon.computeField(pos_data[k, 0:3])).T

            # Compute uncertainty in small body rotating frame
            Ppos_P[k, 0:3, 0:3] = np.matmul(dcm_PN1, np.matmul(Pxx[k, 0:3, 0:3], dcm_PN1.T))
            Pacc_P[k, 0:3, 0:3] = np.matmul(dcm_PN1, np.matmul(Pxx[k, 6:9, 6:9], dcm_PN1.T))

        # Save variables in outputs
        if i == 0:
            # Initialize position, velocity and unmodeled acceleration
            outputs.results.pos_BP_N1 = pos_BP_N1
            outputs.results.vel_BP_N1 = vel_BP_N1
            outputs.results.acc_BP_N1 = acc_BP_N1

            # Initialize gravity data
            outputs.results.pos_data = pos_data
            outputs.results.acc_data = acc_data

            # Initialize covariances
            outputs.results.Pxx = Pxx
            outputs.results.Ppos_P = Ppos_P
            outputs.results.Pacc_P = Pacc_P

            # Save filter times
            #outputs.results.tcpu_dmcukf = tcpu_dmcukf[1:n]
        else:
            # Fill position, velocity and unmodeled acceleration
            outputs.results.pos_BP_N1 = np.concatenate((outputs.results.pos_BP_N1, pos_BP_N1[1:n, :]))
            outputs.results.vel_BP_N1 = np.concatenate((outputs.results.vel_BP_N1, vel_BP_N1[1:n, :]))
            outputs.results.acc_BP_N1 = np.concatenate((outputs.results.acc_BP_N1, acc_BP_N1[1:n, :]))

            # Fill gravity data
            outputs.results.pos_data = np.concatenate((outputs.results.pos_data, pos_data[1:n, :]))
            outputs.results.acc_data = np.concatenate((outputs.results.acc_data, acc_data[1:n, :]))

            # Fill covariances
            outputs.results.Pxx = np.concatenate((outputs.results.Pxx, Pxx[1:n, :, :]))
            outputs.results.Ppos_P = np.concatenate((outputs.results.Ppos_P, Ppos_P[1:n, :, :]))
            outputs.results.Pacc_P = np.concatenate((outputs.results.Pacc_P, Pacc_P[1:n, :, :]))

            # Save dmc-ukf computational times
            #outputs.results.tcpu_dmcukf = np.concatenate((outputs.results.tcpu_dmcukf,
            #                                              tcpu_dmcukf[1:n]))

    # This function prepares data for bsk gravity estimation
    def prepare_masconfit():
        # Extract indexes
        idx = np.where(np.logical_and(t >= parameters.times_dmcukf[i, 0],
                                      t < parameters.times_dmcukf[i, 1]))[0]
        flag_meas = outputs.camera.flag_meas[idx]

        # Preallocate data batch and weights
        t_data = t[idx]
        pos_data = outputs.results.pos_data[idx, 0:3]
        acc_data = outputs.results.acc_data[idx, 0:3]

        # Prune data with available measurements
        t_data = t_data[flag_meas]
        pos_data = pos_data[flag_meas, 0:3]
        acc_data = acc_data[flag_meas, 0:3]

        # Include ejecta if needed
        if parameters.grav_est.flag_ejecta:
            idx = rng.choice(len(pos_data), size=parameters.grav_est.n_ejecta, replace=False)
            pos_data[idx, 0:3] = outputs.groundtruth.pos_EP_P[0, 0:parameters.grav_est.n_ejecta, 0:3]
            acc_data[idx, 0:3] = outputs.groundtruth.acc_EP_P[0, 0:parameters.grav_est.n_ejecta, 0:3]

        return t_data, pos_data, acc_data

    # ------------------- SIMULTANEOUS NAVIGATION AND GRAVITY ESTIMATION ------------------------- #
    # Set random seed
    rng = np.random.default_rng(0)

    # Retrieve ground truth times and training segments
    t = outputs.groundtruth.t
    n_segments = len(parameters.times_groundtruth) - 1

    # Set dmc-ukf initial condition
    pos0 = outputs.groundtruth.pos_BP_N1[0, :]
    vel0 = outputs.groundtruth.vel_BP_N1[0, :]
    acc0 = np.zeros(3)
    parameters.dmcukf.x_k = np.concatenate((pos0, vel0, acc0))

    # Preallocate mascon distribution
    outputs.results.mu_M = np.zeros((parameters.grav_est.n_M+1, n_segments+1))
    outputs.results.pos_M = np.zeros((parameters.grav_est.n_M+1, 3, n_segments+1))

    # Preallocate computational times and loss
    outputs.results.tcpu_gravest = np.zeros(n_segments)
    outputs.results.loss = np.zeros((parameters.grav_est.maxiter, n_segments))

    # Initialize basilisk gravity estimation module
    masconfit_bsk = masconFit.MasconFit()
    initialize_masconfit()

    # Initialize basilisk dmc-ukf module
    dmcukf_bsk = dmcukf.DMCUKF()
    initialize_dmcukf()
    dmcukf_bsk.Reset(0)

    # Loop through simulation segments
    for i in range(n_segments):
        # Print status
        print(str(i) + '/' + str(n_segments) + ' segments completed')

        # Load measurements batch
        t0 = parameters.times_groundtruth[i]
        tf = parameters.times_groundtruth[i+1]
        load_measurements()

        # Reset acceleration to zero after fit
        if i > 0:
            x_k = np.array(dmcukf_bsk.x)
            x_k[6:9] = 0
            dmcukf_bsk.x = x_k.tolist()

        # Run filter forward
        dmcukf_bsk.processBatch()

        # Save data
        save_dmcukf()

        # Prepare data for gravity estimation
        t_data, r_data, a_data = prepare_masconfit()

        # Do gravity fit
        masconfit_bsk.train(r_data.tolist(), a_data.tolist())

        # Save loss history
        outputs.results.loss[:, i] = np.array(masconfit_bsk.getLoss()).squeeze()
        #outputs.results.tcpu_gravest[i] = gravest_bsk.tcpu

        # Set gravity variables into dmc-ukf
        dmcukf_bsk.setMascon(masconfit_bsk.muM, masconfit_bsk.posM)

        # Fill output variables
        outputs.results.mu_M[:, i+1] = np.array(masconfit_bsk.muM).squeeze()
        outputs.results.pos_M[:, 0:3, i+1] = np.array(masconfit_bsk.posM).squeeze()

        # Reset some mascon to the unity
        muMtemp = np.array(masconfit_bsk.muM)
        muMtemp[muMtemp < 1] = 1
        masconfit_bsk.muM = muMtemp.tolist()

    # Save mascon distribution
    parameters.grav_est.mu_M = np.array(dmcukf_bsk.mascon.muM)
    parameters.grav_est.pos_M = np.array(dmcukf_bsk.mascon.xyzM)

    # Compute inhomogeneous gravity time series
    n_data = len(outputs.results.acc_data)
    outputs.results.accNK_data = outputs.results.acc_data
    for i in range(n_data):
        outputs.results.accNK_data[i, 0:3] -= \
            -parameters.asteroid.mu * outputs.results.pos_data[i, 0:3]\
            / np.linalg.norm(outputs.results.pos_data[i, 0:3])**3
    gravity_series(parameters, outputs)
