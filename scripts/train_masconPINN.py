import numpy as np
import pickle as pck

from src.scenario import Scenario
from plots.plots_gravity import all_gravityplots

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3


# This sets a configuration dict for mascon-PINN training
def configuration():
    # Set configuration file
    config = {
        'groundtruth': {'file': '',  # filled automatically
                        'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': bsk_path + 'eros200700.tab',
                        # 'file_poly': bsk_path + 'eros007790.tab',
                        # 'file_poly': bsk_path + 'ver128q.tab',
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
        'regression': {'file': '',
                       'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                       'grav_model': 'pinn',  # 'mascon' / 'pinn' / 'spherharm'
                       'data': {'n_data': 100000},
                       'grad_descent': {'maxiter': 5000,
                                        'lr': 1e-3,
                                        'batch_size': 2000,
                                        'loss': 'linear'},
                       'pinn': {'n_inputs': 5,
                                'neurons': 40,
                                'layers': 6,
                                'activation': 'SIREN',  # 'GELU' / 'SiLU' / 'SIREN'
                                'r_ad': 3.47 * km2m,
                                'switch': {'rad_bc': 16. * km2m,
                                           'r_bc': 70. * km2m,
                                           'k_bc': 1.,
                                           'l_bc': 1},
                                #'eval_mode': 'BSK',  # 'BSK' / 'autograd'
                                'model_bc': {'file': 'mascon100_muxyz_quadratic_octantrand0.pck',
                                             'type': 'mascon',
                                             'n_M': 100}}}
    }

    return config


# This loads data and train mascon based on config
def launch_training(config):
    config_pinn = config['regression']['pinn']
    config_gd = config['regression']['grad_descent']

    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    gt = scenario.groundtruth
    gt.set_file(config['groundtruth'])
    gt.import_data(n_data=config['regression']['data']['n_data'])

    # Initialize estimation
    scenario.init_regression()
    asteroid = scenario.regression.asteroid

    # Import mascon model
    file_mascon = '/'.join(scenario.regression.file.split('/')[:-1]) \
                  + '/' + config_pinn['model_bc']['file']
    #file_mascon = 'Results/eros/results/polyheterogeneous/ideal/dense_alt50km_100000samples/mascon100_muxyzquadratic_octant_rand0.pck'
    inputs = pck.load(open(file_mascon, "rb"))
    mu_M = inputs.regression.asteroid.gravity[0].mu_M
    xyz_M = inputs.regression.asteroid.gravity[0].xyz_M

    # Add pinn model in training mode
    asteroid.add_pinn(training=True)
    asteroid.add_mascon(mu_M=mu_M, xyz_M=xyz_M)

    # Initialize pinn nn and prepare optimizer
    pinn = asteroid.gravity[0]
    pinn.add_mascon(mu_M, xyz_M)
    pinn.set_extra_params(r_ad=config_pinn['r_ad'],
                          rad_bc=config_pinn['switch']['rad_bc'],
                          r_bc=config_pinn['switch']['r_bc'],
                          k_bc=config_pinn['switch']['k_bc'],
                          l_bc=config_pinn['switch']['l_bc'])
    pinn.init_network(n_inputs=config_pinn['n_inputs'],
                      n_layers=config_pinn['layers'],
                      n_neurons=config_pinn['neurons'],
                      activation=config_pinn['activation'])
    pinn.prepare_optimizer(maxiter=config_gd['maxiter'],
                           lr=config_gd['lr'],
                           batch_size=config_gd['batch_size'],
                           loss_type=config_gd['loss'])

    # # Retrieve data to train
    # pos_data = groundtruth.spacecraft.data.pos_BP_P
    # acc_data = groundtruth.spacecraft.data.acc_BP_P

    # Retrieve data to train
    pos_data = gt.spacecraft.data.pos_BP_P
    acc_data = gt.spacecraft.data.acc_BP_P
    U_data = gt.spacecraft.data.U
    # pos_ejecta = groundtruth.ejecta.data.pos_BP_P
    # acc_ejecta = groundtruth.ejecta.data.acc_BP_P
    # U_ejecta = groundtruth.ejecta.data.U
    #
    # pos_data = np.concatenate((pos_data,
    #                            pos_ejecta))
    # acc_data = np.concatenate((acc_data,
    #                           acc_ejecta))
    # U_data = np.concatenate((U_data,
    #                         U_ejecta))

    # Train pinn
    print('------- Initiating pinn training -------')
    pinn.train(pos_data, acc_data, U_data)
    print('------- Finished pinn training -------')
    pinn.compute_proxy(pos_data)

    # Save pinn model
    pinn.save_model(path=scenario.estimation.file)

    # Create pinn gravity
    pinn.create_gravity()

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
