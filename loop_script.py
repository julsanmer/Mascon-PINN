import numpy as np
import os

from Classes.configClass import Configuration
from Launchers.launcher import launch


def prepare_simulation(dev_lmk=0, maskangle_sun=0):
    # Set asteroid name and ground truth gravity
    configuration.asteroid_name = 'eros'
    configuration.grav_groundtruth = 'poly'

    # Define gravity estimation parameters
    configuration.mascon_type = 'MUPOS'
    configuration.mascon_init = 'octant'
    #configParams.nM = np.array([100,200,300,400,500,600,700,800,900,1000])
    #configuration.nM_array = np.array([100, 500, 1000])
    configuration.nM_array = np.array([200])
    configuration.rand_M = 1

    # Define Adam gradient descent variables
    configuration.maxiter = 1000
    configuration.lr = 1e-3
    configuration.loss_type = 'MSE'

    # Define simulation and training orbits batches
    configuration.orbits_groundtruth = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    configuration.orbits_dmcukf = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                            [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
    maxorbits = 10

    # Define DMC-UKF rate and gravity estimation data sampling
    configuration.dmcukf_rate = 60
    configuration.gravest_rate = 60

    # Define camera parameters
    configuration.f = 25 * 1e-3
    configuration.n_lmk = 100
    configuration.dev_lmk = dev_lmk

    # Define initial orbit p
    configuration.a0 = 34 * 1e3
    configuration.i0 = 45 * np.pi/180

    # Define ejecta properties
    configuration.flag_ejecta = True
    configuration.n_ejecta = 50
    configuration.dev_ejecta = 0.05

    # Define visibility for truth simulation
    configuration.maskangle_sun = maskangle_sun
    if maskangle_sun >= 0:
        visibility_str = 'shadow'
    else:
        visibility_str = 'visible'

    # Create asteroid folder if it does not exist
    path_asteroid = 'Results/' + configuration.asteroid_name
    exist = os.path.exists(path_asteroid)
    if not exist:
        os.makedirs(path_asteroid)

    # Extract data and results type
    data_type = configuration.data_type
    results_type = configuration.results_type

    # Create ground truth path if it does not exist and define ground truth file
    path_groundtruth = path_asteroid + '/groundtruth/' + configuration.grav_groundtruth
    exist = os.path.exists(path_groundtruth)
    if not exist:
        os.makedirs(path_groundtruth)
    if data_type == 'orbit':
        file_groundtruth = path_groundtruth + '/a' + str(int(configuration.a0/1e3)) + 'km' \
                           + 'i' + str(int(configuration.i0 * 180/np.pi)) + 'deg' + '_' \
                           + str(maxorbits) + 'orbits' + '.pck'
    elif data_type == 'dense':
        file_groundtruth = path_groundtruth + '/dense.pck'

    # Create camera path if it does not exist and define camera file
    if results_type == 'simultaneous':
        path_camera = path_asteroid + '/camera' + '/a' + str(int(configuration.a0/1e3)) + 'km' \
                      + 'i' + str(int(configuration.i0 * 180/np.pi)) + 'deg' + '_' + str(maxorbits) + 'orbits'
        exist = os.path.exists(path_camera)
        if not exist:
            os.makedirs(path_camera)
        file_camera = path_groundtruth + '/' + str(int(configuration.f*1e3)) + 'mm' + '_' \
                      + str(configuration.n_lmk) + 'landmarks' + '_' + visibility_str + '.pck'
    elif results_type == 'ideal':
        file_camera = []

    # Define results path
    if results_type == 'simultaneous':
        filepath_results = path_asteroid + '/results/simultaneous' + '/a' + str(int(configuration.a0/1e3)) + 'km'\
                           + 'i' + str(int(configuration.i0*180/np.pi)) + 'deg_' \
                           + str(configuration.orbits_groundtruth[-1]) + 'orbits_' + str(configuration.dmcukf_rate) \
                           + 's' + str(configuration.dev_lmk) + 'm' + str(int(configuration.f*1e3)) + 'mm'
    elif results_type == 'ideal':
        if data_type == 'orbit':
            filepath_results = path_asteroid + '/results/ideal' + '/a' + str(int(configuration.a0/1e3)) \
                           + 'km' + 'i' + str(int(configuration.i0*180/np.pi)) + 'deg_' \
                           + str(configuration.orbits_groundtruth[-1]) + 'orbits_' + str(configuration.dmcukf_rate) \
                           + 's'
        elif data_type == 'dense':
            filepath_results = path_asteroid + '/results/ideal' + '/dense_' + str(configuration.dmcukf_rate) + 's'
    exist = os.path.exists(filepath_results)
    if not exist:
        os.makedirs(filepath_results)

    # Fill ground truth and camera files
    configuration.file_groundtruth = file_groundtruth
    configuration.file_camera = file_camera

    # Fill results path
    configuration.filepath_results = filepath_results


def simulate_groundtruth():
    # Set parameters (TBD)
    configuration.seed_M = 0
    configuration.n_M = 1

    # Launch ground truth simulation
    launch(configuration)


def simulate_camera():
    # Set parameters (TBD)
    configuration.seed_M = 0
    configuration.n_M = 1

    # Launch camera simulation
    launch(configuration)


def mascon_loop():
    # Set strings for output file
    ejecta_str = ''
    if configuration.flag_ejecta:
        ejecta_str = 'ejectaM' + str(configuration.n_ejecta)
    eclipse_str = ''
    if configuration.maskangle_sun >= 0:
        eclipse_str = '_eclipse'

    # Loop changing mascon parameters
    for i in range(configuration.rand_M):
        for j in range(len(configuration.nM_array)):
            # Prepare results file
            configuration.seed_M = i
            configuration.n_M = configuration.nM_array[j]
            file_results = configuration.filepath_results + '/mascon' + str(configuration.n_M) + '_' \
                           + configuration.mascon_type + configuration.loss_type + '_' + configuration.mascon_init \
                           + 'LR1E' + str(int(np.log10(configuration.lr))) + eclipse_str + ejecta_str + '_rand' \
                           + str(i) + '.pck'
            configuration.file_results = file_results

            # Launch simulation
            launch(configuration)


if __name__ == "__main__":
    # Call configuration class
    configuration = Configuration()
    #configuration.data_type = 'dense'
    #configuration.results_type = 'ideal'
    configuration.data_type = 'orbit'
    configuration.results_type = 'simultaneous'

    prepare_simulation(dev_lmk=0, maskangle_sun=-90*np.pi/180)
    configuration.flag_groundtruth = False
    configuration.flag_camera = False
    configuration.flag_results = True
    configuration.flag_plot = True
    configuration.flag_map2D = True
    configuration.flag_map3D = True

    # Do ground truth
    if configuration.flag_groundtruth:
        simulate_groundtruth()

    # Do camera
    if configuration.flag_camera:
        simulate_camera()

    # Loop
    if configuration.flag_results:
        mascon_loop()

        # prepare_simulation(dev_lmk=5, maskangle_sun=-90*np.pi/180)
        # configParams.sim = False
        # configParams.cam = False
        # configParams.nav = True
        # configParams.plots = False
        # configParams.computeGrav2D = True
        # configParams.computeGrav3D = True
        # navloop(configParams)
        #
        # configureSim(configParams, 0, 5*np.pi/180)
        # configParams.sim = False
        # configParams.cam = False
        # configParams.nav = True
        # configParams.plots = False
        # configParams.computeGrav2D = True
        # configParams.computeGrav3D = True
        # navloop(configParams)
        #
        # configureSim(configParams, 5, 5*np.pi/180)
        # configParams.sim = False
        # configParams.cam = False
        # configParams.nav = True
        # configParams.plots = False
        # configParams.computeGrav2D = True
        # configParams.computeGrav3D = True
        # navloop(configParams)
