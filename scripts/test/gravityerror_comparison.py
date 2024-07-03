import numpy as np
import pickle as pck
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rc

# ------ COMPONENTS & SUBPLOT HANDLING ------ #
rad2deg = 180 / np.pi
sec2day = 1.0 / (3600*24)
m2km = 1e-3
m2mm = 1e3
m2mu = 1e6
err2perc = 1e2
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

font = 20
font_legend = 15
font_map = 15
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_gravity3D():
    h_3D = outputs_pinn.groundtruth.gravmap.hXYZ_3D
    aErr3D_pinn = outputs_pinn.results.gravmap.aErrXYZ_3D
    aErr3D_mascon = outputs_mascon.results.gravmap.aErrXYZ_3D
    aErr3D_masconpinv = outputs_masconpinv.results.gravmap.aErrXYZ_3D
    aErr3D_masconnnls = outputs_masconnnls.results.gravmap.aErrXYZ_3D
    aErr3D_masconmu = outputs_masconmu.results.gravmap.aErrXYZ_3D
    a0Err3D = outputs_pinn.groundtruth.gravmap.a0ErrXYZ_3D
    h = outputs_pinn.groundtruth.states.h_BP

    # Preallocate number of data for radius range
    n_truth = 10
    h_min = np.min(h)
    h_max = np.max(h)
    N_truth = np.zeros(n_truth)

    # Obtain number of data in each radius range
    for i in range(n_truth):
        hmin_i = h_min + i * (h_max-h_min) / n_truth
        hmax_i = h_min + (i+1) * (h_max-h_min) / n_truth
        idx = np.where(np.logical_and(h >= hmin_i, h <= hmax_i))[0]
        N_truth[i] = len(idx)
    N_truth /= np.sum(N_truth)

    # Transpose variables to 1D vectors
    h_1D = np.ravel(h_3D)
    aErr1D_pinn = np.ravel(aErr3D_pinn)
    aErr1D_mascon = np.ravel(aErr3D_mascon)
    aErr1D_masconpinv = np.ravel(aErr3D_masconpinv)
    aErr1D_masconnnls = np.ravel(aErr3D_masconnnls)
    aErr1D_masconmu = np.ravel(aErr3D_masconmu)
    a0Err1D_pinn = np.ravel(a0Err3D)

    # Prepare bins for gravity errors
    h0 = np.nanmin(h_1D)
    hf = np.nanmax(h_1D)
    n_bins = 80
    h_lim = np.linspace(h0, hf + 1e-6*hf, n_bins+1)
    h_bins = h_lim[0:-1] + (h_lim[1] - h_lim[0]) / 2
    aErrbins_pinn = np.zeros(n_bins)
    aErrbins_mascon = np.zeros(n_bins)
    aErrbins_masconpinv = np.zeros(n_bins)
    aErrbins_masconnnls = np.zeros(n_bins)
    aErrbins_masconmu = np.zeros(n_bins)
    a0Errbins_pinn = np.zeros(n_bins)
    N_bins = np.zeros(n_bins)

    # Compute average gravity error in each radius range
    for i in range(len(h_1D)):
        if np.isnan(h_1D[i]):
            continue
        dh = h_1D[i] - h0
        idx = int(np.floor(dh / (h_lim[1]-h_lim[0])))
        aErrbins_pinn[idx] += aErr1D_pinn[i]
        aErrbins_mascon[idx] += aErr1D_mascon[i]
        aErrbins_masconpinv[idx] += aErr1D_masconpinv[i]
        aErrbins_masconnnls[idx] += aErr1D_masconnnls[i]
        aErrbins_masconmu[idx] += aErr1D_masconmu[i]
        a0Errbins_pinn[idx] += a0Err1D_pinn[i]
        N_bins[idx] += 1
    aErrbins_pinn /= N_bins
    aErrbins_mascon /= N_bins
    aErrbins_masconpinv /= N_bins
    aErrbins_masconnnls /= N_bins
    aErrbins_masconmu /= N_bins
    a0Errbins_pinn /= N_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot 3D gravity error and their average
    markersize = 2
    alpha = 0.05
    ax.plot(h_1D*m2km, aErr1D_pinn*err2perc, color=colors[0], alpha=alpha,
            marker='o', linestyle='', markersize=markersize)
    ax.plot(h_1D*m2km, aErr1D_mascon*err2perc, color=colors[1], alpha=alpha,
            marker='o', linestyle='', markersize=markersize)
    ax.plot(h_1D*m2km, aErr1D_masconmu*err2perc, color=colors[2], alpha=alpha,
            marker='o', linestyle='', markersize=markersize)
    ax.plot(h_1D*m2km, aErr1D_masconpinv*err2perc, color=colors[3], alpha=alpha,
            marker='o', linestyle='', markersize=markersize)
    ax.plot(h_1D*m2km, aErr1D_masconnnls*err2perc, color=colors[4], alpha=alpha,
            marker='o', linestyle='', markersize=markersize)

    ax.plot(np.nan, np.nan, color=colors[0], alpha=1, marker='o', linestyle='',
            markersize=5, label='100 mascon (Adam) + 8x40 PINN SIREN')
    ax.plot(np.nan, np.nan, color=colors[1], alpha=1, marker='o', linestyle='',
            markersize=5, label='100 mascon (Adam)')
    ax.plot(np.nan, np.nan, color=colors[2], alpha=1, marker='o', linestyle='',
            markersize=5, label='100 mascon (Adam), only $\mu_M$ train')
    ax.plot(np.nan, np.nan, color=colors[3], alpha=1, marker='o', linestyle='',
            markersize=5, label='100 mascon (pinv) w. $\mu_M<0$ allowed')
    ax.plot(np.nan, np.nan, color=colors[4], alpha=1, marker='o', linestyle='',
            markersize=5, label='100 mascon (nnls) w. $\mu_M>0$')

    ax.plot(h_bins*m2km, aErrbins_pinn*err2perc, linewidth=2, color=colors[5])
    ax.plot(h_bins*m2km, aErrbins_mascon*err2perc, linewidth=2, color=colors[6])
    ax.plot(h_bins*m2km, aErrbins_masconmu*err2perc, linewidth=2, color=colors[7])
    ax.plot(h_bins*m2km, aErrbins_masconpinv*err2perc, linewidth=2, color=colors[8])
    ax.plot(h_bins*m2km, aErrbins_masconnnls*err2perc, linewidth=2, color=colors[9])

    # Plot data lines
    plt.axvline(x=np.min(h)*m2km, color='b')
    plt.axvline(x=np.max(h)*m2km, color='b')
    for i in range(n_truth):
        ax.bar(h_min*m2km + (i+0.5)*(h_max-h_min)/n_truth*m2km, 1e3*N_truth[i],
               (h_max-h_min)/n_truth*m2km, 0, alpha=0.2, color='red')

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    plt.xlabel('Altitude [km]', fontsize=font)
    plt.ylabel('Gravity error [\%]', fontsize=font)

    # Set limits on axis
    ax.set_xlim([np.nanmin(h_1D)*m2km, np.nanmax(h_1D)*m2km])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)

# Load files
file_path = os.path.dirname(os.getcwd())
file_pinn = file_path + \
            '/Results/eros/results/poly/ideal/dense_alt50km_10000samples/' +\
            'pinn8x40SIREN_masconMLE.pck'
file_mascon = file_path +\
              '/Results/eros/results/poly/ideal/dense_alt50km_10000samples/' +\
              'mascon100_muxyzMLE_octant_rand0.pck'
file_masconpinv = file_path +\
              '/Results/eros/results/poly/ideal/dense_alt50km_10000samples/' +\
              'mascon100_pinv_octant_rand0.pck'
file_masconnnls = file_path +\
              '/Results/eros/results/poly/ideal/dense_alt50km_10000samples/' +\
              'mascon100_nnls_octant_rand0.pck'
file_masconmu = file_path +\
              '/Results/eros/results/poly/ideal/dense_alt50km_10000samples/' +\
              'mascon100_muMLE_octant_rand0.pck'

# Extract orbits class
parameters, outputs_pinn = pck.load(open(file_pinn, "rb"))
_, outputs_mascon = pck.load(open(file_mascon, "rb"))
_, outputs_masconpinv = pck.load(open(file_masconpinv, "rb"))
_, outputs_masconnnls = pck.load(open(file_masconnnls, "rb"))
_, outputs_masconmu = pck.load(open(file_masconmu, "rb"))

# Plot orbits and RMSE
plot_gravity3D()
#plot_orbits()
#plot_RMSE()
plt.show()
