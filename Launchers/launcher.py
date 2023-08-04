import pickle

from Launchers.camera import run_camera
from Launchers.dmcukf import run_dmcukf
from Launchers.groundtruth import ScenarioAsteroid, run_groundtruth
from Classes.scenarioClass import Parameters, Outputs
from Plotting.plots_dmcukf import *
from Gravity.gravity_map import gravity_map2D, gravity_map3D, gravity_series
from Gravity.mascon_features import train_mascon


def launch(configuration):
    def load_groundtruth():
        # Import simulation data
        _, inputs = pickle.load(open(configuration.file_groundtruth, "rb"))

        if configuration.data_type == 'orbit':
            # Prune times
            idxSim = np.linspace(0, np.floor((tf - t0) / dt) * dt,
                                 int(np.floor((tf - t0) / dt)) + 1).astype(int)

            outputs.groundtruth = inputs.groundtruth

            # Prune simulation variables
            outputs.groundtruth.t = inputs.groundtruth.t[idxSim]
            outputs.groundtruth.pos_BP_N0 = inputs.groundtruth.pos_BP_N0[idxSim, :]
            outputs.groundtruth.vel_BP_N0 = inputs.groundtruth.vel_BP_N0[idxSim, :]
            outputs.groundtruth.pos_BP_N1 = inputs.groundtruth.pos_BP_N1[idxSim, :]
            outputs.groundtruth.vel_BP_N1 = inputs.groundtruth.vel_BP_N1[idxSim, :]
            outputs.groundtruth.pos_BP_P = inputs.groundtruth.pos_BP_P[idxSim, :]
            outputs.groundtruth.vel_BP_P = inputs.groundtruth.vel_BP_P[idxSim, :]
            outputs.groundtruth.acc_BP_P = inputs.groundtruth.acc_BP_P[idxSim, :]
            outputs.groundtruth.accHigh_BP_P = inputs.groundtruth.accHigh_BP_P[idxSim, :]
            outputs.groundtruth.r_BP = inputs.groundtruth.r_BP[idxSim]
            outputs.groundtruth.h_BP = inputs.groundtruth.h_BP[idxSim]

            n_ejecta = parameters.grav_est.n_ejecta
            n_segments, _, _ = inputs.groundtruth.pos_EP_P.shape
            dev_ejecta = parameters.grav_est.dev_ejecta
            if parameters.grav_est.flag_ejecta:
                err_ejecta = 1 + np.random.multivariate_normal(np.zeros(3), (dev_ejecta ** 2) * np.identity(3),
                                                               (n_segments, n_ejecta))
                outputs.groundtruth.pos_EP_P = inputs.groundtruth.pos_EP_P[:, 0:n_ejecta, 0:3]
                outputs.groundtruth.acc_EP_P = inputs.groundtruth.acc_EP_P[:, 0:n_ejecta, 0:3] * err_ejecta
                outputs.groundtruth.r_EP = inputs.groundtruth.r_EP[:, 0:n_ejecta]
                outputs.groundtruth.h_EP = inputs.groundtruth.h_EP[:, 0:n_ejecta]

            outputs.groundtruth.pos_PS_N1 = inputs.groundtruth.pos_PS_N1[idxSim, :]
            outputs.groundtruth.e_SP_P = inputs.groundtruth.e_SP_P[idxSim, :]
            outputs.groundtruth.mrp_PN0 = inputs.groundtruth.mrp_PN0[idxSim, :]

            outputs.groundtruth.mrp_BP = inputs.groundtruth.mrp_BP[idxSim, :]
        elif configuration.data_type == 'dense':
            # Prune times
            #idxSim = np.where(np.logical_and(outputs.groundtruth.t >= t0, outputs.groundtruth.t <= tf))[0]
            # idxSim = np.linspace(0, np.floor((tf - t0) / dt) * dt, int(np.floor((tf - t0) / dt)) + 1).astype(int)
            # print(idxSim)

            outputs.groundtruth = inputs.groundtruth

            # Prune simulation variables
            #outputs.groundtruth.t = inputs.groundtruth.t[idxSim]
            #outputs.groundtruth.r_CA_A = inputs.groundtruth.r_CA_A[idxSim, :]
            #outputs.groundtruth.a_A = inputs.groundtruth.a_A[idxSim, :]

    # Launch the specified simulation
    if configuration.flag_groundtruth:
        # Initialize ground truth parameters and initialize outputs
        parameters = Parameters(configuration)
        outputs = Outputs()

        # Create the scenario
        scenario = ScenarioAsteroid(parameters)

        # Launch the scenario
        run_groundtruth(scenario, parameters, outputs)

        # Compute truth gravity maps
        gravity_map2D(parameters, outputs, 'groundtruth')
        gravity_map3D(parameters, outputs, 'groundtruth')

        # Save simulation outputs
        with open(configuration.file_groundtruth, "wb") as f:
            pickle.dump([parameters, outputs], f)
    elif configuration.flag_camera:
        # Initialize camera parameters and initialize outputs
        parameters = Parameters(configuration)
        outputs = Outputs()

        # Import ground truth data
        t0 = parameters.times_groundtruth[0]
        tf = parameters.times_groundtruth[-1]
        dt = parameters.save_rate
        load_groundtruth()

        # Launch camera simulation
        run_camera(parameters, outputs)

        # Save simulation outputs
        with open(configuration.file_camera, "wb") as f:
            pickle.dump([parameters, outputs], f)
    else:
        # Initialize DMC-UKF parameters and initialize outputs
        parameters = Parameters(configuration)
        outputs = Outputs()

        # Import ground truth and camera data
        t0 = parameters.times_groundtruth[0]
        tf = parameters.times_groundtruth[-1]

        if configuration.results_type == 'simultaneous':
            dt = int(parameters.dmcukf_rate / parameters.save_rate)
            load_groundtruth()
            load_camera(t0, tf, dt, outputs, configuration.file_camera)

            # Launch simultaneous dmc-ukf and mascon gravity estimation
            run_dmcukf(parameters, outputs)
        elif configuration.results_type == 'ideal':
            dt = int(parameters.dmcukf_rate)
            load_groundtruth()

            # Launch gravity fit
            train_mascon(parameters, outputs)

        # Compute 2D gravity error maps
        if configuration.flag_map2D:
            gravity_map2D(parameters, outputs, 'results')

        # Compute 3D gravity error map
        if configuration.flag_map3D:
            gravity_map3D(parameters, outputs, 'results')

        # Save simulation outputs
        with open(configuration.file_results, "wb") as f:
            pickle.dump([parameters, outputs], f)

        # Show plots if required
        if configuration.flag_plot:
            launch_plots(configuration, parameters, outputs)


def load_camera(t0, tf, dt, outputs, file_camera):
    # Import simulation data
    _, inputs = pickle.load(open(file_camera, "rb"))

    # Prune times
    idx = np.linspace(0, np.floor((tf-t0)/dt)*dt, int(np.floor((tf-t0)/dt))+1).astype(int)

    # Copy camera variables
    outputs.camera.t = inputs.camera.t[idx]
    outputs.camera.pixel_lmk = inputs.camera.pixel_lmk[idx, :]
    outputs.camera.nvisible_lmk = inputs.camera.nvisible_lmk[idx]
    outputs.camera.isvisible_lmk = inputs.camera.isvisible_lmk[idx, :]
    outputs.camera.flag_meas = inputs.camera.flag_meas[idx]
    outputs.camera.mrp_CP = inputs.camera.mrp_CP[idx, :]


def launch_plots(configuration, parameters, outputs):
    if parameters.grav_est.data_type == 'orbit':
        # Plot orbits in small body centred inertial and fixed frames
        plot_orb(parameters, outputs, frame='inertial')
        plot_orb(parameters, outputs, frame='asteroid')

        # Plot truth and estimated positions, velocities and non-Keplerian accelerations
        plot_pos(outputs.groundtruth.t, outputs.groundtruth.pos_BP_N1,
                 outputs.results.pos_BP_N1, outputs.camera.flag_meas)
        plot_vel(outputs.groundtruth.t, outputs.groundtruth.vel_BP_N1,
                 outputs.results.vel_BP_N1, outputs.camera.flag_meas)
        plot_acc(outputs.groundtruth.t, outputs.results.accNK_poly,
                 outputs.results.accNK_data, outputs.camera.flag_meas)

        # Plot mascon distribution
        plot_mascon(parameters.grav_est.pos_M, parameters.grav_est.mu_M,
                    parameters.asteroid.xyz_vert, parameters.asteroid.order_face)

        # Plot errors in positions, velocities and non-Keplerian accelerations
        plot_pos_error(outputs.groundtruth.t, np.subtract(outputs.results.pos_BP_N1, outputs.groundtruth.pos_BP_N1),
                       outputs.results.Pxx, outputs.camera.flag_meas)
        plot_vel_error(outputs.groundtruth.t, np.subtract(outputs.results.vel_BP_N1, outputs.groundtruth.vel_BP_N1),
                       outputs.results.Pxx, outputs.camera.flag_meas)

    # Plot global gravity maps
    if configuration.flag_map2D:
        plot_gravity2D(outputs, parameters)
    if configuration.flag_map3D:
        plot_gravity3D(outputs, parameters)

    # Show all plots
    plt.show()
