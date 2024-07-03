import pickle as pck

from plots.plots_gravity import all_gravityplots


# Path and file
path = '/Users/julio/Desktop/python_scripts/THOR/scripts/'
#file = 'Results/eros/results/poly/ideal/dense_alt50km_100000samples/mascon100_muxyzMSE_octant_rand0.pck'
#file = 'Results/eros/results/polyheterogeneous/ideal/dense_alt50km_100000samples/pinn6x160SIREN_masconMLE.pck'
file = 'Results/eros/results/polyheterogeneous/ideal/dense_alt50km_10000samples/mascon100_muxyzMSE_octant_rand0.pck'


# Load scenario
scenario = pck.load(open(path + file, "rb"))

# Do plots
all_gravityplots(scenario)
