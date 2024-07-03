import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
from matplotlib import rc

from src.spaceObjects.asteroid import Asteroid
from src.maneuvers.transfers import Transfers
from scripts.transfer import solve_transfer, simulate_truth

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

km2m = 1e3
m2km = 1e-3
h2sec = 3600
deg2rad = np.pi/180
rad2deg = 180/np.pi

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
font = 20


# This function creates initial positions using an orbit
def initial_position(oe, mu, angvel, n_nu=1):
    # Create a grid of true anomalies
    nu = np.linspace(0, 2*np.pi, n_nu)

    # Retrieve orbital elements
    a, e, inc, omega, RAAN = \
        oe[0], oe[1], oe[2], oe[3], oe[4]

    # Compute orbital angular velocity
    n = np.sqrt(mu / a**3)

    # Compute latitude and longitude
    u = omega + nu
    t = n * nu
    lat = np.arcsin(np.sin(u) * np.sin(inc))
    lon = RAAN + angvel*t + np.arctan2(np.sin(u) * np.cos(inc),
                                       np.cos(u))

    # Sort longitude
    idx = lon.argsort()
    lon, lat = lon[idx], lat[idx]

    # Fill positions
    pos = a * np.array([np.cos(lon) * np.cos(lat),
                        np.sin(lon) * np.cos(lat),
                        np.sin(lat)]).T

    return pos

# This adds gravity models to asteroid
def copy_gravity(asteroid, asteroid_orig):
    # Loop through models
    for model in asteroid_orig.gravity:
        # If model is mascon
        if model.name == 'mascon':
            # Retrieve mascon distribution
            mu_M, xyz_M = model.mu_M, model.xyz_M

            # Add mascon
            asteroid.add_mascon(mu_M, xyz_M)
        # If model is pinn
        elif model.name == 'pinn':
            # Add pinn model
            asteroid.add_pinn(model.file_torch)


# Set configuration file
config = {
    'groundtruth': {'asteroid_name': 'eros',  # 'eros'
                    'grav_model': 'poly',
                    #'file_poly': bsk_path + 'eros007790.tab',
                    'file_poly': bsk_path + 'eros200700.tab',
                    'n_eval': 100},
    'estimation': {'t_span': [0, 4*h2sec],
                   'pos0': np.array([-34*km2m, 0, 0]),
                   'posf': np.array([0, -4*km2m, 0]),
                   #'grav_model': ['mascon'],
                   #'grav_model': ['mascon', 'pinn'],
                   #'main_path': 'Results/eros/results/poly200700faces/ideal/dense_alt50km_100000samples/',
                   #'file': 'mascon1000_muxyz_quadratic_octantrand0',
                   'model_path': '/ideal/dense_alt50km_100000samples/',
                   'file': 'pinn6x40SIREN_linear_mascon100'}
                   #'file_model': ['mascon_models/mascon100_muxyz_quadratic_octantrand0.pck']}
                   #'file_model': ['mascon_models/mascon100_muxyz_quadratic_octantrand0.pck',
                   #               'pinn_models/pinn6x40SIREN_linear_mascon100.pt']}
}

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# File
file = 'Results/eros/results/poly200700faces' \
        + config['estimation']['model_path'] + config['estimation']['file'] + '.pck'
scenario = pck.load(open(file, "rb"))

# Retrieve asteroid parameters
mu = scenario.groundtruth.asteroid.mu
angvel = 2*np.pi / scenario.groundtruth.asteroid.rot_period

# Create initial conditions
oe0 = np.array([34*km2m,
                0,
                0*deg2rad,
                0*deg2rad,
                0*deg2rad])
n_nu = 5
pos0 = initial_position(oe0, mu, angvel, n_nu=n_nu)

# Target position and arrival time
posf = np.array([0, -3.75, 0]) * km2m
posf = np.tile(posf, (n_nu, 1))
tf = 4*h2sec
tf = np.repeat(tf, n_nu)

# Create asteroid for transfer
asteroid = scenario.estimation.asteroid
for model in asteroid.gravity:
    model.create_gravity()
asteroid.shape.create_shape()

# Create transfers instance and solve them
transfers = Transfers(asteroid, pos0, posf, tf)
transfers.solve_transfers()

# Create asteroid for truth
asteroid_truth = scenario.groundtruth.asteroid
for model in asteroid_truth.gravity:
    model.create_gravity()

# Compute transfers result
transfers.asteroid_truth = asteroid_truth
transfers.propagate_transfers()


# Load gravity models
i = 0
for model in scenario.estimation.asteroid.gravity_names:
    # Add corresponding model
    if name == 'mascon':
        # Import mascon and add to asteroid
        mu_M, xyz_M = 4, 4
        asteroid.add_mascon(mu_M=mascon_dict['mu_M'],
                            xyz_M=mascon_dict['xyz_M'])
    elif model == 'pinn':
        # Import pinn and add to asteroid
        asteroid.add_pinn(file_torch=config['transfer']['main_path']
                                     + config['transfer']['file_model'][i])

# Outputs list
pos_list = []
success_list = []
hascollide_list = []
errposf_list = []

# Train
for i in range(len(lon0)):

    config['transfer']['pos0'] = pos0
    config['transfer']['posf'] = posf
    config['transfer']['t_span'][1] = tf[i]

    # Launch solver
    sol = solve_transfer(config)

    if not sol.has_collide and sol.success:
        # Launch dynamics
        asteroid, t, pos, vel = simulate_truth(config, sol)

        # Compute final error
        errpos_f = np.linalg.norm(pos[-1, 0:3] - posf)
    else:
        pos = []
        errpos_f = np.nan

    # Append
    success_list.append(sol.success)
    hascollide_list.append(sol.has_collide)
    pos_list.append(pos)
    errposf_list.append(errpos_f)

# Prepare for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
pos0 = config['transfer']['pos0']
posf = config['transfer']['posf']
for i in range(len(lon0)):
    if not hascollide_list[i] and success_list[i]:
        pos = pos_list[i]
        plt.plot(pos[0, 0]*m2km, pos[0, 1]*m2km, pos[0, 2]*m2km, marker='.')
        plt.plot(pos[:, 0]*m2km, pos[:, 1]*m2km, pos[:, 2]*m2km)

# Plot asteroid
ax.plot_trisurf(asteroid.shape.xyz_vert[:, 0]*m2km,
                asteroid.shape.xyz_vert[:, 1]*m2km,
                asteroid.shape.xyz_vert[:, 2]*m2km,
                triangles=asteroid.shape.order_face-1,
                color=[105/255, 105/255, 105/255],
                zorder=0,
                alpha=.2)
ax.axis('equal')

# Prepare for error plotting
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)

# Plot
ax.plot(lon0*rad2deg, np.array(errposf_list), marker='.')

# Set labels
ax.set_xlabel(r'$\theta$ [$^{\circ}$]', fontsize=font)
ax.set_ylabel(r'$\delta r_f$ [m]', fontsize=font)

# Set ticks, grid and legend
ax.tick_params(axis='both', labelsize=font)
ax.grid()
plt.tight_layout()

plt.show()
