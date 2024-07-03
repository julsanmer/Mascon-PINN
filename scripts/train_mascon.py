import numpy as np
import pickle as pck

from src.scenario import Scenario

from plots.plots_gravity import all_gravityplots

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3


# This sets a configuration dict for mascon training
def configuration():
    # Set configuration file
    config = {
        'groundtruth': {'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': bsk_path + 'eros200700.tab',
                        #'file_poly': bsk_path + 'eros007790.tab',
                        #'file_poly': bsk_path + 'ver128q.tab',
                        'n_face': [],  # to be filled later
                        'data': 'dense',  # 'dense' / 'orbit'
                        'mascon': {'add': False,
                                   'mu_M': np.array([0.1, -0.1]) * 4.46275472004 * 1e5,
                                   'xyz_M': np.array([[10, 0, 0],
                                                      [-10, 0, 0]]) * 1e3},
                        'dense': {'dist': 'alt',  # 'alt', 'rad', 'ell'
                                  'rmax': 50 * km2m,
                                  'n_data': 100000},
                        'gravmap': {'nr_3D': 40, 'nlat_3D': 40, 'nlon_3D': 40, 'rmax_3D': 160 * km2m,
                                    'n_2D': 160, 'rmax_2D': 60 * km2m}},
        'estimation': {'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                       'grav_model': 'mascon',  # 'mascon' / 'pinn' / 'spherharm'
                       'data': {'dmcukf_rate': 60,
                                'grav_rate': 60,
                                'orbits': np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                                    [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]),
                                'add_ejecta': False,  # bool
                                'n_ejecta': 50,
                                'dev_ejecta': 0,
                                'n_data': 100000},
                       'grad_descent': {'maxiter': 5000,
                                        'lr': 1e-3,
                                        'batch_size': 10000,
                                        'loss': 'quadratic',
                                        'train_xyz': True},
                       'mascon': {'train_xyz': True,  # bool
                                  'init': 'octant',
                                  'n_M': 100,
                                  'seed_M': 0}}
    }

    return config


# This loads data and train mascon based on config
def launch_training(config):
    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    groundtruth = scenario.groundtruth
    groundtruth.set_file(config['groundtruth'])
    groundtruth.import_data(n_data=config['estimation']['data']['n_data'])

    # Initialize estimation
    scenario.init_estimation()
    asteroid = scenario.estimation.asteroid

    # for i in range(len(asteroid.shape.normal_face)):
    #     n_face = asteroid.shape.normal_face[i, 0:3]
    #     xyz_face = asteroid.shape.xyz_face[i, 0:3]
    #
    #     test = xyz_face + n_face*10
    #     flag = asteroid.shape.check_exterior(test)
    #     if not flag:
    #         print('aaaa')
    #     #angle = np.dot(n_face, xyz_face) / np.linalg.norm(xyz_face)
    #     #if angle < 0:
    #     #    print(xyz_face / np.linalg.norm(xyz_face))
    #     #    print(n_face)

    # Add mascon model in training mode
    asteroid.add_mascon(training=True)
    mascon = asteroid.gravity[0]

    # Set mascon initial distribution
    mascon.init_distribution(asteroid,
                             config['estimation']['mascon']['n_M'])

    # Prepare mascon optimizer
    config_gd = config['estimation']['grad_descent']
    mascon.prepare_optimizer(loss=config_gd['loss'],
                             maxiter=config_gd['maxiter'],
                             lr=config_gd['lr'],
                             batch_size=config_gd['batch_size'],
                             xyzM_ad=np.array([16342.6, 8410.61, 5973.615])/10,
                             muM_ad=asteroid.mu / (config['estimation']['mascon']['n_M']+1),
                             train_xyz=config_gd['train_xyz'])

    # Retrieve data to train
    pos_data = groundtruth.spacecraft.data.pos_BP_P
    acc_data = groundtruth.spacecraft.data.acc_BP_P
    # pos_ejecta = groundtruth.ejecta.data.pos_BP_P
    # acc_ejecta = groundtruth.ejecta.data.acc_BP_P
    #
    # pos_data = np.concatenate((pos_data,
    #                            pos_ejecta))
    # acc_data = np.concatenate((acc_data,
    #                           acc_ejecta))

    # Train mascon distribution
    print('------- Initiating mascon fit -------')
    mascon.train(pos_data, acc_data)
    mascon.delete_optimizer()
    print('------- Finished mascon fit -------')

    # Save mascon model
    mascon.save_model(path=scenario.estimation.file)

    # Create mascon gravity
    mascon.create_gravity()

    # Import gravity map grids
    gravmap = scenario.estimation.gravmap
    gravmap.import_grids(groundtruth.gravmap)

    # Generate gravity maps and errors
    gravmap.generate_maps(asteroid)
    gravmap.error_maps(groundtruth.gravmap)

    # Delete swigpy objects
    asteroid.delete_swigpy()

    # Save simulation outputs
    with open(scenario.estimation.file, "wb") as f:
        pck.dump(scenario, f)

    return scenario


if __name__ == "__main__":
    # Generate configuration
    config = configuration()

    # Launch training
    scenario = launch_training(config)

    # Plot gravity
    all_gravityplots(scenario)
