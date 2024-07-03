import numpy as np

from Basilisk.fswAlgorithms import masconFit
from Basilisk.utilities import unitTestSupport as sp
from Basilisk.utilities import RigidBodyKinematics as rbk

from Basilisk.ExternalModules import dmcukf

from src.gravitymodels.mascon.initializers import initialize_mascon
from src.gravitymodels.gravity_map import gravity_series


# This function executes the simultaneous navigation
# and gravity estimation algorithms
def run_dmcukf(parameters, outputs):
    # This function initializes bsk dmc-ukf properties
    def initialize_dmcukf():
        # Set dmc-ukf mascon model
        dmcukf_bsk.setMascon(mascon_params.mu_M0.tolist(),
                             mascon_params.xyz_M0.tolist())

        # Set SRP and Sun's 3rd body if required
        if parameters.flag_sun:
            dmcukf_bsk.useSRP = True
            dmcukf_bsk.useSun = True
            dmcukf_bsk.m = spacecraft_params.mass
            dmcukf_bsk.ASRP = spacecraft_params.srpArea
            dmcukf_bsk.CR = spacecraft_params.CR

        # Set dmc-ukf state, uncertainty, process and measurement covariances
        dmcukf_bsk.x = sp.np2EigenVectorXd(dmcukf_params.x_k + dmcukf_params.dx_k)
        dmcukf_bsk.Pxx = dmcukf_params.P_k.tolist()
        dmcukf_bsk.Pproc = dmcukf_params.P_proc.tolist()
        dmcukf_bsk.Ppixel = dmcukf_params.P_pixel.tolist()

        # Set integration steps
        dmcukf_bsk.Nint = dmcukf_params.n_prop

        # Set landmark and camera information
        dmcukf_bsk.nLmk = len(sensors_params.xyz_lmk)
        dmcukf_bsk.rLmk = sensors_params.xyz_lmk + sensors_params.dxyz_lmk
        dmcukf_bsk.f = sensors_params.f
        dmcukf_bsk.wPixel = sensors_params.w_pixel

        # Set dcm of planet equatorial w.r.t. inertial and
        # camera w.r.t. body frame
        dmcukf_bsk.dcm_N1N0 = parameters.asteroid.dcm_N1N0
        dmcukf_bsk.dcm_CB = sensors_params.dcm_CB

    # This function initializes bsk gravity estimation properties
    def initialize_masconfit():
        # Set gravity estimation algorithm parameters
        masconfit_bsk.nM = int(mascon_params.n_M) + 1

        # Set adimensional variables
        masconfit_bsk.mu = mascon_params.mu
        masconfit_bsk.muMad = mascon_params.muM_ad
        masconfit_bsk.xyzMad = mascon_params.xyzM_ad.tolist()

        # Choose loss function type
        masconfit_bsk.useMSE = True
        masconfit_bsk.useMLE = False

        # Set training variables flag
        if mascon_params.train_xyz:
            masconfit_bsk.trainXYZ = True
        else:
            masconfit_bsk.trainXYZ = False

        # Set Adam parameters
        masconfit_bsk.setMaxIter(graddescent_params.maxiter)
        masconfit_bsk.setLR(graddescent_params.lr)

        # Set polyhedron
        shape = masconfit_bsk.shapeModel
        shape.xyzVertex = mascon_params.xyz_vert.tolist()
        shape.orderFacet = mascon_params.order_face.tolist()

        # Initialize mascon distribution
        xyz_M0, mu_M0 = initialize_mascon(masconfit_bsk, mascon_params)
        mascon_params.xyz_M0 = xyz_M0
        mascon_params.mu_M0 = mu_M0
        masconfit_bsk.xyzM = xyz_M0.tolist()
        masconfit_bsk.muM = mu_M0.tolist()

    # This function loads measurements into bsk dmc-ukf module
    def load_measurements():
        # Extract corresponding batch of measurements
        idx = np.where(np.logical_and(t >= t0, t <= tf))[0]
        if i > 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        t_meas = t[idx].tolist()

        # Extract small body position and orientation
        pos_PS_N1 = states_truth.pos_PS_N1[idx, :].tolist()
        mrp_PN0 = states_truth.mrp_PN0[idx, :].tolist()
        mrp_BP = states_truth.mrp_BP[idx, :].tolist()

        # Load small body ephemeris and orientation
        dmcukf_bsk.batch.r_PS_N1 = pos_PS_N1
        dmcukf_bsk.batch.mrp_PN0 = mrp_PN0

        # Loop spacecraft orientation w.r.t. small body rotating frame
        dmcukf_bsk.batch.mrp_BP = mrp_BP

        # Extract camera variables
        pixel_lmk = meas_camera.pixel_lmk[idx, :].tolist()
        isvisible_lmk = meas_camera.isvisible_lmk[idx, :].tolist()
        mrp_CP = meas_camera.mrp_CP[idx, :].tolist()

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
            states_results.pos_BP_N1 = pos_BP_N1
            states_results.vel_BP_N1 = vel_BP_N1
            states_results.acc_BP_N1 = acc_BP_N1

            # Initialize gravity data
            states_results.pos_data = pos_data
            states_results.acc_data = acc_data

            # Initialize covariances
            states_results.Pxx = Pxx
            states_results.Ppos_P = Ppos_P
            states_results.Pacc_P = Pacc_P
        else:
            # Fill position, velocity and unmodeled acceleration
            states_results.pos_BP_N1 = np.concatenate((states_results.pos_BP_N1,
                                                       pos_BP_N1[1:n, :]))
            states_results.vel_BP_N1 = np.concatenate((states_results.vel_BP_N1,
                                                       vel_BP_N1[1:n, :]))
            states_results.acc_BP_N1 = np.concatenate((states_results.acc_BP_N1,
                                                       acc_BP_N1[1:n, :]))

            # Fill gravity data
            states_results.pos_data = np.concatenate((states_results.pos_data, pos_data[1:n, :]))
            states_results.acc_data = np.concatenate((states_results.acc_data, acc_data[1:n, :]))

            # Fill covariances
            states_results.Pxx = np.concatenate((states_results.Pxx, Pxx[1:n, :, :]))
            states_results.Ppos_P = np.concatenate((states_results.Ppos_P, Ppos_P[1:n, :, :]))
            states_results.Pacc_P = np.concatenate((states_results.Pacc_P, Pacc_P[1:n, :, :]))

    # This function prepares data for bsk gravity estimation
    def prepare_masconfit():
        # Extract indexes
        idx = np.where(np.logical_and(t >= parameters.times_dmcukf[i, 0],
                                      t < parameters.times_dmcukf[i, 1]))[0]
        flag_meas = outputs.camera.flag_meas[idx]

        # Preallocate data batch and weights
        t_data = t[idx]
        pos_data = states_results.pos_data[idx, 0:3]
        acc_data = states_results.acc_data[idx, 0:3]

        # Prune data with available measurements
        t_data = t_data[flag_meas]
        pos_data = pos_data[flag_meas, 0:3]
        acc_data = acc_data[flag_meas, 0:3]

        # Include ejecta if needed
        if parameters.grav_est.flag_ejecta:
            idx = rng.choice(len(pos_data), size=parameters.grav_est.n_ejecta, replace=False)
            pos_data[idx, 0:3] = states_truth.pos_EP_P[0, 0:parameters.grav_est.n_ejecta, 0:3]
            acc_data[idx, 0:3] = states_truth.acc_EP_P[0, 0:parameters.grav_est.n_ejecta, 0:3]

        return t_data, pos_data, acc_data

    # ------------------- SIMULTANEOUS NAVIGATION AND GRAVITY ESTIMATION ------------------------- #
    # Get different parameters
    dmcukf_params = parameters.dmcukf
    graddescent_params = parameters.grav_est.grad_descent
    mascon_params = parameters.grav_est.mascon
    sensors_params = parameters.sensors
    spacecraft_params = parameters.spacecraft

    # Set random seed
    rng = np.random.default_rng(0)

    # Set object to save results
    states_truth = outputs.groundtruth.states
    meas_camera = outputs.measurements
    states_results = outputs.results.states

    # Retrieve ground truth times and training segments
    t = states_truth.t
    n_segments = len(parameters.times_groundtruth) - 1

    # Set dmc-ukf initial condition
    pos0 = states_truth.pos_BP_N1[0, :]
    vel0 = states_truth.vel_BP_N1[0, :]
    acc0 = np.zeros(3)
    dmcukf_params.x_k = np.concatenate((pos0, vel0, acc0))

    # Preallocate mascon distribution
    states_results.mu_M = np.zeros((mascon_params.n_M+1, n_segments+1))
    states_results.xyz_M = np.zeros((mascon_params.n_M+1, 3, n_segments + 1))

    # Preallocate computational times and loss
    states_results.tcpu_gravest = np.zeros(n_segments)
    states_results.loss = np.zeros((graddescent_params.maxiter, n_segments))

    # Initialize basilisk gravity estimation module
    masconfit_bsk = masconFit.MasconFit()
    initialize_masconfit()

    # Initialize basilisk dmc-ukf module
    dmcukf_bsk = dmcukf.DMCUKF()
    initialize_dmcukf()
    dmcukf_bsk.Reset(0)

    # Initialize training
    print('------- Initiating simultaneous navigation '
          'and gravity estimation -------')

    # Loop through simulation segments
    for i in range(n_segments):
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
        masconfit_bsk.train(r_data.tolist(), a_data.tolist(), False)

        # Save loss history
        states_results.loss[:, i] = np.array(masconfit_bsk.getLoss()).squeeze()

        # Set gravity variables into dmc-ukf
        dmcukf_bsk.setMascon(masconfit_bsk.muM, masconfit_bsk.xyzM)

        # Fill output variables
        states_results.mu_M[:, i+1] = np.array(masconfit_bsk.muM).squeeze()
        states_results.xyz_M[:, 0:3, i+1] = np.array(masconfit_bsk.xyzM).squeeze()

        # Print status
        print(str(i+1) + '/' + str(n_segments) + ' segments completed')

    # Finish training
    print('------- Finished simultaneous navigation '
          'and gravity estimation -------')

    # Save mascon distribution
    mascon_params.mu_M = np.array(dmcukf_bsk.mascon.muM)
    mascon_params.xyz_M = np.array(dmcukf_bsk.mascon.xyzM)

    # Compute inhomogeneous gravity time series
    n_data = len(states_results.acc_data)
    states_results.accNK_data = states_results.acc_data
    for i in range(n_data):
        states_results.accNK_data[i, 0:3] -= \
            -parameters.asteroid.mu * states_results.pos_data[i, 0:3]\
            / np.linalg.norm(states_results.pos_data[i, 0:3])**3
    gravity_series(parameters, outputs)
