import numpy as np
import matplotlib.pyplot as plt
import pickle as pck

from src.maneuvers.solver import Solver
from src.maneuvers.propagator import Propagator
from src.spaceObjects.asteroid import Asteroid


from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3
m2km = 1e-3
h2sec = 3600


# This sets a configuration dict for transfer
def configuration():
    # Set configuration file
    config = {
        'groundtruth': {'asteroid_name': 'eros',  # 'eros'
                        'grav_model': 'poly',
                        'file_poly': bsk_path + 'eros200700.tab',
                        'n_eval': 100},
        'transfer': {'t_span': (0, 4*h2sec),
                     'pos0': np.array([-34*km2m, 0, 0]),
                     'posf': np.array([0, -4*km2m, 0]),
                     #'grav_model': ['mascon'],
                     'grav_model': ['mascon', 'pinn'],
                     'main_path': 'Results/eros/results/poly200700faces/ideal/dense_alt50km_100000samples/',
                     'file_model': ['mascon_models/mascon100_muxyz_quadratic_octantrand0.pck',
                                    'pinn_models/pinn6x40SIREN_linear_mascon100.pt']}
                     #'file_model': ['mascon_models/mascon100_muxyz_quadratic_octantrand0.pck']}
    }

    return config


# This function solves transfer
def solve_transfer(config):
    # Create asteroid
    asteroid = Asteroid()
    asteroid.set_properties(config['groundtruth']['asteroid_name'],
                            file_shape=config['groundtruth']['file_poly'])

    # Load gravity models
    i = 0
    for model in config['transfer']['grav_model']:
        # Add corresponding model
        if model == 'mascon':
            # Import mascon and add to asteroid
            mascon_dict = pck.load(open(config['transfer']['main_path']
                                        + config['transfer']['file_model'][i],
                                        "rb"))
            asteroid.add_mascon(mu_M=mascon_dict['mu_M'],
                                xyz_M=mascon_dict['xyz_M'])
        elif model == 'pinn':
            # Import pinn and add to asteroid
            asteroid.add_pinn(file_torch=config['transfer']['main_path']
                                         + config['transfer']['file_model'][i])

        # Add
        i += 1

    # Retrieve bc
    t0 = config['transfer']['t_span'][0]
    tf = config['transfer']['t_span'][1]
    pos0 = config['transfer']['pos0']
    posf = config['transfer']['posf']

    # Create solver object
    transfer = Transfer()

    # Fill solver object
    transfer.asteroid = asteroid
    transfer.t0, transfer.tf = t0, tf
    transfer.pos0 = pos0
    transfer.posf = posf
    transfer.t_ad = tf - t0
    transfer.r_ad = np.linalg.norm(posf - pos0)

    # Solve
    sol = transfer.solve_bvp()

    # Add dimensional factors to sol
    sol.t_ad = transfer.t_ad
    sol.r_ad = transfer.r_ad

    # Check collisions
    pos = sol.y[0:3, :].T * sol.r_ad
    has_collide = transfer.check_collision(pos)
    sol.has_collide = has_collide

    return sol


# This function simulates transfer
def simulate_truth(config, sol):
    # Get initial conditions
    t0 = config['transfer']['t_span'][0]
    tf = config['transfer']['t_span'][1]
    pos0 = config['transfer']['pos0']
    vel0 = sol.y[3:6, 0] * (sol.r_ad/sol.t_ad)

    # True asteroid
    asteroid = Asteroid()
    asteroid.set_properties(config['groundtruth']['asteroid_name'],
                            file_shape=config['groundtruth']['file_poly'])
    asteroid.add_poly(config['groundtruth']['file_poly'])

    # Propagator
    propagator = Propagator()
    propagator.atol, propagator.rtol = 1e-9, 1e-9
    propagator.n_eval = config['groundtruth']['n_eval']
    propagator.asteroid = asteroid

    # Propagate
    t, pos, vel = propagator.propagate((t0, tf),
                                        pos0,
                                        vel0)

    return asteroid, t, pos, vel


if __name__ == "__main__":
    # Retrieve configuration
    config = configuration()

    # Solve transfer
    sol = solve_transfer(config)

    # Dynamics
    asteroid, t, pos, vel = simulate_truth(config, sol)

    # Prepare for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    pos0 = config['transfer']['pos0']
    posf = config['transfer']['posf']
    plt.plot(pos[:, 0]*m2km, pos[:, 1]*m2km, pos[:, 2]*m2km)
    plt.plot(pos0[0]*m2km, pos0[1]*m2km, pos0[2]*m2km,
             marker='s')
    plt.plot(posf[0]*m2km, posf[1]*m2km, posf[2]*m2km,
             marker='o')

    # Plot asteroid
    ax.plot_trisurf(asteroid.shape.xyz_vert[:, 0]*m2km,
                    asteroid.shape.xyz_vert[:, 1]*m2km,
                    asteroid.shape.xyz_vert[:, 2]*m2km,
                    triangles=asteroid.shape.order_face-1,
                    color=[105/255, 105/255, 105/255],
                    zorder=0,
                    alpha=.2)
    ax.axis('equal')

    plt.show()
