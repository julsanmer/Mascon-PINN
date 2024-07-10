import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
from matplotlib import rc

from src.spaceObjects.asteroid import Asteroid
from src.maneuvers.transfers import Transfers

from Basilisk.simulation import gravityEffector
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
                    'grav_model': 'polyheterogeneous',
                    #'file_poly': bsk_path + 'eros007790.tab',
                    'file_poly': bsk_path + 'eros200700.tab',
                    'n_eval': 100},
    'estimation': {'model_path': '/ideal/dense_alt50km_100000samples/',
                   #'file': 'pinn6x40SIREN_linear_mascon100'
                   'file': 'mascon100_muxyz_quadratic_octantrand0'
                   },
    'transfer': {'t_span': [0, 4*h2sec],
                 'oe0': np.array([34*km2m, 0, 0*deg2rad, 0*deg2rad, 0*deg2rad]),
                 'n_nu': 40,
                 'posf': np.array([0, -4, 0]) * km2m}
}

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Set type of transfer
#transfer_name = 'polar'
transfer_name = 'equatorial'
if transfer_name == 'equatorial':
    # Set initial inclination and final position
    config['transfer']['oe0'][2] = 0 * deg2rad
    config['transfer']['posf'] = np.array([0, -3.75, 0]) * km2m
elif transfer_name == 'polar':
    # Set initial inclination and final position
    config['transfer']['oe0'][2] = 90 * deg2rad
    config['transfer']['posf'] = np.array([0, 0, 6]) * km2m

# Obtain number of faces
config_gt = config['groundtruth']
config_est = config['estimation']
_, _, _, n_face = \
    gravityEffector.loadPolyFromFileToList(config_gt['file_poly'])

# File
model_gt = config_gt['grav_model'] + str(n_face) + 'faces'
file = 'Results/eros/results/' + model_gt + '/' \
       + config_est['model_path'] + config_est['file'] + '.pck'
scenario = pck.load(open(file, "rb"))

# Retrieve asteroid parameters
mu = scenario.groundtruth.asteroid.mu
angvel = 2*np.pi / scenario.groundtruth.asteroid.rot_period

# Create initial conditions
oe0 = config['transfer']['oe0']
n_nu = config['transfer']['n_nu']
posf = config['transfer']['posf']
tf = config['transfer']['t_span'][1]
pos0 = initial_position(oe0, mu, angvel, n_nu=n_nu)
posf = np.tile(posf, (n_nu, 1))
tf = np.repeat(tf, n_nu)

# Retrieve gravity parameters
gravity = scenario.estimation.asteroid.gravity

# Create asteroid for transfer with only mascon
asteroid = Asteroid()
asteroid.set_properties('eros',
                        file_shape=config['groundtruth']['file_poly'])
for i in range(len(gravity)):
    if gravity[i].name == 'mascon':
        asteroid.add_mascon(gravity[i].mu_M,
                            gravity[i].xyz_M)

# Create transfers instance and solve them
transfers = Transfers(asteroid, pos0, posf, tf)
transfers.solve_transfers()

# Add pinn to previous asteroid and solve
for i in range(len(gravity)):
    if gravity[i].name == 'pinn':
        asteroid.add_pinn(file_torch=gravity[0].file_torch)
        transfers.asteroid = asteroid
        transfers.solve_transfers()

# Create asteroid for truth analysis
asteroid_truth = scenario.groundtruth.asteroid
for model in asteroid_truth.gravity:
    if model.name == 'poly':
        model.file = config['groundtruth']['file_poly']
    model.create_gravity()

# Compute transfers result
transfers.asteroid_truth = asteroid_truth
transfers.propagate_transfers()

# Delete swigpy objects
transfers.asteroid.delete_swigpy()
transfers.asteroid_truth.delete_swigpy()

# Save transfers
file_save = 'Results/eros/results/' + model_gt + '/' \
            + config['estimation']['model_path'] + config['estimation']['file'] \
            + '_transfers' + transfer_name + '.pck'

# Save simulation outputs
with open(file_save, "wb") as f:
    pck.dump(transfers, f)
