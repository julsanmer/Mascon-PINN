import numpy as np
import pickle as pck
import matplotlib.pyplot as plt

from src.scenario import Scenario
from plots.plots_gravity import all_gravityplots

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3


# This sets a configuration dict for mascon-PINN training
def configuration():
    # Set configuration file
    config = {
        'flag_groundtruth': False,  # bool
        'flag_measurements': False,  # bool
        'flag_results': True,  # bool
        'flag_plots': True,  # bool

        'groundtruth': {'file': '',  # filled automatically
                        'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': bsk_path + 'eros007790.tab',
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
        'estimation': {'file': '',
                       'algorithm': 'ideal',  # 'simultaneous' / 'ideal'
                       'grav_model': 'pinn',  # 'mascon' / 'pinn' / 'spherharm'
                       'data': {'n_data': 10000},
                       'grad_descent': {'maxiter': 5000,
                                        'lr': 1e-3,
                                        'loss': 'MLE'},  # 'MSE' / 'MLE'
                       'pinn': {'neurons': 160,
                                'layers': 8,
                                'activation': 'SIREN',  # 'GELU' / 'SiLU' / 'SIREN'
                                'r_ad': 3.47 * km2m,
                                'switch': {'rad_bc': 16. * km2m,
                                           'r_bc': 100. * km2m,
                                           'k_bc': 1.,
                                           'l_bc': 0},
                                'model_bc': {'file': 'mascon100_muxyzMSE_octant_rand0.pck',
                                             'type': 'mascon',
                                             'n_M': 100}}}
    }

    return config


if __name__ == "__main__":
    # Generate configuration
    config = configuration()
    config_pinn = config['estimation']['pinn']
    config_gd = config['estimation']['grad_descent']

    # Create scenario instance
    scenario = Scenario(config)

    # Import groundtruth
    groundtruth = scenario.groundtruth
    groundtruth.set_file(config['groundtruth'])
    groundtruth.file = '/Users/julio/Desktop/python_scripts/THOR/scripts/' \
                       + groundtruth.file
    groundtruth.import_data(n_data=100000)

    # Initialize estimation
    scenario.init_estimation()
    asteroid = scenario.estimation.asteroid

    # Import mascon model
    file_mascon = '/'.join(scenario.estimation.file.split('/')[:-1]) \
                  + '/' + config_pinn['model_bc']['file']
    file_mascon = '/Users/julio/Desktop/python_scripts/THOR/scripts/' \
                  + file_mascon
    inputs = pck.load(open(file_mascon, "rb"))
    mu_M = inputs.estimation.asteroid.gravity[0].mu_M
    xyz_M = inputs.estimation.asteroid.gravity[0].xyz_M

    # Add pinn model in training mode
    asteroid.add_pinn(training=True)
    asteroid.add_mascon(mu_M=mu_M, xyz_M=xyz_M)

    # Initialize pinn nn and prepare optimizer
    pinn = asteroid.gravity[0]
    # mu_M, xyz_M = np.array([asteroid.mu]), np.array([[0, 0, 0]])
    pinn.add_mascon(mu_M, xyz_M)
    pinn.set_extra_params(r_ad=config_pinn['r_ad'],
                          rad_bc=config_pinn['switch']['rad_bc'],
                          r_bc=config_pinn['switch']['r_bc'],
                          k_bc=config_pinn['switch']['k_bc'],
                          l_bc=config_pinn['switch']['l_bc'])
    pinn.init_network(n_layers=config_pinn['layers'],
                      n_neurons=config_pinn['neurons'],
                      activation=config_pinn['activation'])
    pinn.prepare_optimizer(maxiter=config_gd['maxiter'],
                           lr=config_gd['lr'])

    # Retrieve data to train
    pos_data = groundtruth.spacecraft.data.pos_BP_P
    acc_data = groundtruth.spacecraft.data.acc_BP_P
    U_data = groundtruth.spacecraft.data.U
    r_data = groundtruth.spacecraft.data.r_BP
    h_data = groundtruth.spacecraft.data.h_BP

    # Mascon potential
    U_mascon = pinn.bc_potential(pos_data)
    U_pm = groundtruth.asteroid.mu/np.linalg.norm(pos_data, axis=1)

    # # Normalize potential
    #print(pinn.rad_bc)
    #plt.plot(r_data, U_data, marker='.', linestyle='')
    plt.figure()
    plt.plot(r_data/pinn.rad_bc, abs(U_mascon - U_data)/U_data*100,
             marker='.', linestyle='', label='Mascon n=100')
    plt.plot(r_data/pinn.rad_bc, abs(U_pm - U_data)/U_data*100,
             marker='.', linestyle='', label='Point mass')
    plt.xlabel('r/R [-]')
    plt.ylabel('Potential error [\%]')
    plt.semilogy()
    plt.legend()
    # plt.plot(pinn.r_data / pinn.rad_bc, U_data - pinn.U_bc,
    #          marker='.', linestyle='')

    plt.figure()
    plt.plot(r_data, U_mascon, marker='.', linestyle='', zorder=10)
    plt.plot(r_data, U_data, marker='.', linestyle='')

    # Generate some sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(-x)

    dU0 = U_mascon - U_data
    dU1 = (U_mascon - U_data) * (r_data/pinn.rad_bc)
    dU2 = (U_mascon - U_data) * (r_data / pinn.rad_bc)**2
    dU3 = (U_mascon - U_data) * (r_data / pinn.rad_bc)**3

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot data in the first subplot
    axs[0, 0].plot(r_data/pinn.rad_bc, dU0/np.max(dU0), marker='.', linestyle='', zorder=10)
    axs[0, 0].set_title('$l_{bc}$=0')
    axs[0, 0].set_ylabel('$U_{proxy}$ [-]')

    # Plot data in the second subplot
    axs[0, 1].plot(r_data/pinn.rad_bc, dU1 / np.max(dU1), marker='.', linestyle='', zorder=10)
    axs[0, 1].set_title('$l_{bc}$=1')

    # Plot data in the third subplot
    axs[1, 0].plot(r_data/pinn.rad_bc, dU2 / np.max(dU2), marker='.', linestyle='', zorder=10)
    axs[1, 0].set_title('$l_{bc}$=2')
    axs[1, 0].set_xlabel('$r/R$ [-]')
    axs[1, 0].set_ylabel('$U_{proxy}$ [-]')

    # Plot data in the fourth subplot
    axs[1, 1].plot(r_data/pinn.rad_bc, dU3 / np.max(dU3), marker='.', linestyle='', zorder=10)
    axs[1, 1].set_title('$l_{bc}$=3')
    axs[1, 1].set_xlabel('$r/R$ [-]')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
