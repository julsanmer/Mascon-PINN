import pickle

from Launchers.camera import run_camera
from Launchers.dmcukf import run_dmcukf
from Launchers.groundtruth import ScenarioAsteroid, run_groundtruth
from Classes.scenarioClass import Parameters, Outputs
from Plotting.plots_dmcukf import *
from Gravity.gravity_map import gravity_map2D, gravity_map3D
from Gravity.mascon_features import train_mascon


def launch(configuration):
    # Load ground truth data
    def load_groundtruth():
        # Import simulation data
        _, inputs = pickle.load(open(configuration.file_groundtruth, "rb"))

        if configuration.data_type == 'orbit':
            # Prune times
            # idxSim = np.where(np.logical_and(outputs.groundtruth.t >= t0, outputs.groundtruth.t <= tf))[0]
            idxSim = np.linspace(0, np.floor((tf - t0) / dt) * dt, int(np.floor((tf - t0) / dt)) + 1).astype(int)
            # print(idxSim)

            outputs.groundtruth = inputs.groundtruth

            # Prune simulation variables
            outputs.groundtruth.t = inputs.groundtruth.t[idxSim]
            outputs.groundtruth.pos_CA_N = inputs.groundtruth.pos_CA_N[idxSim, :]
            outputs.groundtruth.vel_CA_N = inputs.groundtruth.vel_CA_N[idxSim, :]
            outputs.groundtruth.pos_CA_A = inputs.groundtruth.pos_CA_A[idxSim, :]
            outputs.groundtruth.vel_CA_A = inputs.groundtruth.vel_CA_A[idxSim, :]
            outputs.groundtruth.acc_CA_N = inputs.groundtruth.acc_CA_N[idxSim, :]
            outputs.groundtruth.acc_CA_A = inputs.groundtruth.acc_CA_A[idxSim, :]
            outputs.groundtruth.accHigh_CA_A = inputs.groundtruth.accHigh_CA_A[idxSim, :]
            outputs.groundtruth.r_CA = inputs.groundtruth.r_CA[idxSim]
            outputs.groundtruth.h_CA = inputs.groundtruth.h_CA[idxSim]

            n_ejecta = parameters.grav_est.n_ejecta
            n_segments, _, _ = inputs.groundtruth.pos_EA_A.shape
            dev_ejecta = parameters.grav_est.dev_ejecta
            if parameters.grav_est.flag_ejecta:
                err_ejecta = 1 + np.random.multivariate_normal(np.zeros(3), (dev_ejecta ** 2) * np.identity(3),
                                                               (n_segments, n_ejecta))
                outputs.groundtruth.pos_EA_A = inputs.groundtruth.pos_EA_A[:, 0:n_ejecta, 0:3]
                outputs.groundtruth.acc_EA_A = inputs.groundtruth.acc_EA_A[:, 0:n_ejecta, 0:3] * err_ejecta
                outputs.groundtruth.r_EA = inputs.groundtruth.r_EA[:, 0:n_ejecta]
                outputs.groundtruth.h_EA = inputs.groundtruth.h_EA[:, 0:n_ejecta]

            outputs.groundtruth.pos_AS_N = inputs.groundtruth.pos_AS_N[idxSim, :]
            outputs.groundtruth.e_SA_A = inputs.groundtruth.e_SA_A[idxSim, :]
            outputs.groundtruth.eul323_AN = inputs.groundtruth.eul323_AN[idxSim, :]
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
            # Compute gravity map
            gravity_map2D(parameters, outputs, 'results')
            #plot_gravity2D(outputs, parameters)
            #plt.show()

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
    outputs.camera.pixel = inputs.camera.pixel[idx, :]
    outputs.camera.n_visible = inputs.camera.n_visible[idx]
    outputs.camera.flag_nav = inputs.camera.flag_nav[idx]
    outputs.camera.latlon = inputs.camera.latlon[idx, :]


def launch_plots(configuration, parameters, outputs):
    # Plot orbits in small body centred inertial and fixed frames
    plot_orb(parameters, outputs, frame='inertial')
    plot_orb(parameters, outputs, frame='asteroid')

    # Plot truth and estimated positions, velocities and non-Keplerian accelerations
    plot_pos(outputs.groundtruth.t, outputs.groundtruth.pos_CA_N, outputs.results.pos_CA_N,
             outputs.camera.flag_nav)
    plot_vel(outputs.groundtruth.t, outputs.groundtruth.vel_CA_N, outputs.results.vel_CA_N,
             outputs.camera.flag_nav)

    #accMod = outputs.results.a_data
    #for i in range(len(aMod)):
    #     aMod[i, 0:3] += parameters.asteroid.mu * np.array(outputs.groundtruth.r_CA_A[i, 0:3]) \
    #                     / np.linalg.norm(outputs.groundtruth.r_CA_A[i, 0:3])**3
    #plot_acc(outputs.groundtruth.t, outputs.groundtruth.a_A, aMod, outputs.camera.flag_nav)

    # Plot mascon distribution
    plot_mascon(parameters.grav_est.pos_M, parameters.grav_est.mu_M,
                parameters.asteroid.xyz_vert, parameters.asteroid.order_face)

    # Plot errors in positions, velocities and non-Keplerian accelerations
    plot_pos_error(outputs.groundtruth.t, np.subtract(outputs.results.pos_CA_N, outputs.groundtruth.pos_CA_N),
                   outputs.results.P, outputs.camera.flag_nav)
    plot_vel_error(outputs.groundtruth.t, np.subtract(outputs.results.vel_CA_N, outputs.groundtruth.vel_CA_N),
                   outputs.results.P, outputs.camera.flag_nav)

    # Plot global gravity maps
    if configuration.flag_map2D:
        plot_gravity2D(outputs, parameters)
    if configuration.flag_map3D:
        plot_gravity3D(outputs.groundtruth.hXYZ_3D, outputs.results.aErrXYZ_3D,
                       outputs.groundtruth.a0ErrXYZ_3D, outputs.groundtruth.h_CA)

    # Show all plots
    plt.show()
