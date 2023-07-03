import pickle

import sys
sys.path.append('/Users/julio/Desktop/python_scripts/testing/modelComparisonAA')
from modelClassBSK import ModelParameters

from Plotting.plots_dmcukf import *
from Plotting.generalPlots import *

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from PIL import Image


def plotDMCUKF():
    font = 22
    # Viz palette
    colors = ['#ffd700',
              '#ffb14e',
              '#fa8775',
              '#ea5f94',
              '#cd34b5',
              '#9d02d7',
              '#0000ff']
    colorsGray = ['#FFFFFF',
                  '#B2BEB5',
                  '#676767']

    """Plot the position estimation error and associated covariance."""
    fileMascon = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMascon, "rb"))
    time = navOutputsM.sim.t
    r_err = navOutputsM.nav.rEst_CA_A - navOutputsM.sim.rTruth_CA_A
    Ppos_A = navOutputsM.nav.Ppos_A
    P = navOutputsM.nav.P
    navCamSolutionRaw = navOutputsM.cam.navMeas

    navCamSolution = np.ones(len(navCamSolutionRaw))
    navCamSolution[navCamSolutionRaw == 0] = np.nan
    tfillRaw = []
    yfillSupRaw = []
    yfillLowRaw = []
    switch = 1
    for ii in range(len(navCamSolutionRaw)):
        if switch == 1 and abs(navCamSolutionRaw[ii]) < 1e-6:
            switch = 0
            t0 = time[ii]
        if switch == 0 and abs(navCamSolutionRaw[ii]-1) < 1e-6:
            tf = time[ii]
            switch = 1
            tfillRaw.append([t0,tf])
            yfillSupRaw.append([10000,10000])
            yfillLowRaw.append([-10000,-10000])
    tfill = np.array(tfillRaw)
    yfillSup = np.array(yfillSupRaw)
    yfillLow = np.array(yfillLowRaw)

    plt.gcf()
    fig1, ax1 = plt.subplots(3, sharex=True, figsize=(12,8))
    fig1.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax1[0].plot(time/(3600*24), r_err[:,0]*navCamSolution, color=colors[6], marker='.',markersize=3,linestyle='')
    ax1[0].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,0,0])*navCamSolution, 'k',linewidth=1)
    ax1[0].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,0,0])*navCamSolution, 'k',linewidth=1)

    ax1[1].plot(time/(3600*24), r_err[:,1]*navCamSolution, color=colors[6], marker='.',markersize=3,linestyle='')
    ax1[1].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,1,1])*navCamSolution, 'k', linewidth=1)
    ax1[1].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,1,1])*navCamSolution, 'k',linewidth=1)

    ax1[2].plot(time/(3600*24), r_err[:,2]*navCamSolution, color=colors[6], marker='.',markersize=3,linestyle='')
    ax1[2].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,2,2])*navCamSolution, 'k',linewidth=1)
    ax1[2].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,2,2])*navCamSolution, 'k',linewidth=1)

    ax1[0].plot(-1e3, 0, color=colors[6],linewidth=1,marker='.', markersize=6, linestyle='',label='Error')
    ax1[0].plot(-1e3, 0, color='black', linewidth=1, label='3-$\sigma$')

    for ii in range(len(tfill)):
        if ii == 0:
            ax1[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                                alpha=0.5, zorder=20)
        else:
            ax1[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                                alpha=0.5, zorder=20)
        ax1[0].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color=colorsGray[2], alpha=0.5,
                            zorder=20)
        ax1[2].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color=colorsGray[2], alpha=0.5,
                            zorder=20)

    ax1[2].set_xlabel('Time [days]', fontsize=font, labelpad=10)

    ax1[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax1[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax1[2].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])

    ax1[0].set_ylim([-30,30])
    ax1[1].set_ylim([-30,30])
    ax1[2].set_ylim([-30,30])

    ax1[0].set_ylabel('$\delta x$ [m]', fontsize=font)
    ax1[1].set_ylabel('$\delta y$ [m]', fontsize=font)
    ax1[2].set_ylabel('$\delta z$ [m]', fontsize=font)

    ax1[0].tick_params(axis='both', labelsize=font)
    ax1[1].tick_params(axis='both', labelsize=font)
    ax1[2].tick_params(axis='both', labelsize=font)

    ax1[0].set_yticks([-30,-15,0,15,30])
    ax1[1].set_yticks([-30,-15,0,15,30])
    ax1[2].set_yticks([-30,-15,0,15,30])

    ax1[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax1[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax1[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax1[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax1[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax1[2].yaxis.set_minor_locator(AutoMinorLocator())

    ax1[0].set_title('$n=100$',fontsize=font)

    ax1[0].grid()
    ax1[1].grid()
    ax1[2].grid()

    ax1[0].legend(fontsize=font-8, ncols=2, bbox_to_anchor=(0.77,1.01),borderaxespad=0.)

    plt.tight_layout()
    #fig.savefig('samplefigure', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
    plt.savefig('Plots/AA/posError.pdf', format='pdf')

    aTruth = navOutputsM.sim.aTruth_A
    aEst = navOutputsM.nav.aTrain
    Pacc_A = navOutputsM.nav.Pacc_A
    for ii in range(len(aEst)):
        aEst[ii, 0:3] += navParamsM.smallbody.mu * np.array(navOutputsM.sim.rTruth_CA_A[ii, 0:3]) / np.linalg.norm(
            navOutputsM.sim.rTruth_CA_A[ii, 0:3])**3
    #for ii in range(len(aTruth)):
    #    aTruth[ii,0:3] -= navParamsM.smallbody.mu * np.array(navOutputsM.sim.rTruth_CA_A[ii, 0:3]) / np.linalg.norm(
    #        navOutputsM.sim.rTruth_CA_A[ii, 0:3])**3
    #errpos = np.sqrt(r_err[:,0]**2+r_err[:,1]**2+r_err[:,2]**2)
    #erracc = np.sqrt((aTruth[:,0]-aEst[:,0])**2 + (aTruth[:,1]-aEst[:,1])**2 + (aTruth[:,2]-aEst[:,2])**2)
    #normacc = np.sqrt(aTruth[:,0]**2 + aTruth[:,1]**2 + aTruth[:,2]**2)
    #print(np.nansum(errpos*navCamSolution)/np.nansum(navCamSolution))
    #print(np.nansum(erracc/normacc*navCamSolution)/np.nansum(navCamSolution)*100)
    plt.gcf()
    fig2, ax2 = plt.subplots(3, sharex=True, figsize=(12,8))
    fig2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax2[0].plot(time/(3600*24), aTruth[:,0]*1e6*navCamSolution, color=colors[1],linewidth=2,zorder=11,alpha=1)
    ax2[1].plot(time/(3600*24), aTruth[:,1]*1e6*navCamSolution, color=colors[1], linewidth=2,zorder=11,alpha=1)
    ax2[2].plot(time/(3600*24), aTruth[:,2]*1e6*navCamSolution, color=colors[1],linewidth=2,zorder=11,alpha=1)

    ax2[0].plot(time/(3600*24), aEst[:,0]*1e6*navCamSolution, color=colors[6], marker='.', markersize=4, linestyle='',
                zorder=10, alpha=1)
    ax2[1].plot(time/(3600*24), aEst[:,1]*1e6*navCamSolution, color=colors[6], marker='.', markersize=4, linestyle='',
                zorder=10, alpha=1)
    ax2[2].plot(time/(3600*24), aEst[:,2]*1e6*navCamSolution, color=colors[6], marker='.', markersize=4, linestyle='',
                zorder=10, alpha=1)

    ax2[0].plot(-1e3, 0, color=colors[1], linewidth=1, label='Truth')
    ax2[0].plot(-1e3, 0, color=colors[6], marker='.', markersize=6, linestyle='', label='DMC-UKF')
    for ii in range(len(tfill)):
        if ii == 0:
            ax2[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                                alpha=.5)
        else:
            ax2[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                                alpha=.5, zorder=20)
        ax2[0].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                            alpha=.5, zorder=20)
        ax2[2].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color=colorsGray[2],
                            alpha=.5, zorder=20)

    plt.xlabel('Time [days]', fontsize=font, labelpad=10)

    ax2[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax2[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax2[2].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])

    ax2[0].set_ylim([-100,100])
    ax2[1].set_ylim([-100,100])
    ax2[2].set_ylim([-100,100])

    ax2[0].set_ylabel('$a_{x}$ [$\mu$m/s$^2$]', fontsize=font)
    ax2[1].set_ylabel('$a_{y}$ [$\mu$m/s$^2$]', fontsize=font)
    ax2[2].set_ylabel('$a_{z}$ [$\mu$m/s$^2$]', fontsize=font)

    ax2[0].tick_params(axis='both', labelsize=font)
    ax2[1].tick_params(axis='both', labelsize=font)
    ax2[2].tick_params(axis='both', labelsize=font)

    ax2[0].set_yticks([-100,-50,0,50,100])
    ax2[1].set_yticks([-100,-50,0,50,100])
    ax2[2].set_yticks([-100,-50,0,50,100])

    ax2[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax2[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax2[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax2[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax2[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax2[2].yaxis.set_minor_locator(AutoMinorLocator())

    ax2[0].set_title('$n=100$',fontsize=font)

    ax2[0].grid()
    ax2[1].grid()
    ax2[2].grid()

    ax2[0].legend(fontsize=font-8, ncols=2, bbox_to_anchor=(0.3,1.19),borderaxespad=0.)
    plt.tight_layout()

    plt.savefig('Plots/AA/accError.pdf', format='pdf')

def plotOnline():
    font = 22
    # Viz palette
    colors = ['#ffd700',
              '#ffb14e',
              '#fa8775',
              '#ea5f94',
              '#cd34b5',
              '#9d02d7',
              '#0000ff']

    plt.gcf()
    fig = plt.figure(figsize=(7.3,6))
    ax = fig.add_subplot(1,1,1)

    # Declare files, open and load
    filePath1 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s0m25mm'
    filePath2 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm'
    fileRef = '/Users/julio/Desktop/python_scripts/testing/modelComparisonAA/results/orbita034kmi045deg/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    fileMUA1 = filePath1 + '/mascon100_MUMSE_octantLR1E-3_rand0.pck'
    fileMUPOSA1 = filePath1 + '/mascon100_MUPOSMSE_octantLR1E-3_rand0.pck'
    fileMUA2 = filePath2 + '/mascon100_MUMSE_octantLR1E-3_rand0.pck'
    fileMUPOSA2 = filePath2 + '/mascon100_MUPOSMSE_octantLR1E-3_rand0.pck'
    fileMUB1 = filePath1 + '/mascon100_MUMSE_octantLR1E-3_eclipse_rand0.pck'
    fileMUPOSB1 = filePath1 + '/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    fileMUB2 = filePath2 + '/mascon100_MUMSE_octantLR1E-3_eclipse_rand0.pck'
    fileMUPOSB2 = filePath2 + '/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'

    mascon, data = pickle.load(open(fileRef, "rb"))
    aErrIdeal = mascon.accErrXYZ3DIntervalAlt
    aErr0 = data.accErrXYZ3DkepIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUA1, "rb"))
    h = navOutputsM.nav.hXYZ3DInterval
    aErrMUA1 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUPOSA1, "rb"))
    aErrMUPOSA1 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUA2, "rb"))
    aErrMUA2 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUPOSA2, "rb"))
    aErrMUPOSA2 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUB1, "rb"))
    aErrMUB1 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUPOSB1, "rb"))
    aErrMUPOSB1 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUB2, "rb"))
    aErrMUB2 = navOutputsM.nav.aErrXYZ3DIntervalAlt
    navParamsM, navOutputsM = pickle.load(open(fileMUPOSB2, "rb"))
    aErrMUPOSB2 = navOutputsM.nav.aErrXYZ3DIntervalAlt

    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErr0[0:-1]*1e2, color='k', zorder=10,linewidth=2)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrIdeal[0:-1]*1e2, color=colors[6], linestyle='--', zorder=10,linewidth=2)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUA1[0:-1]*1e2, color=colors[1], marker='.', clip_on=False,
            zorder=10,markersize=9)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUPOSA1[0:-1]*1e2, color=colors[6], marker='.', clip_on=False,
            zorder=10,markersize=9)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUA2[0:-1]*1e2, color=colors[1], marker='X', clip_on=False,
            zorder=10)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUPOSA2[0:-1]*1e2, color=colors[6], marker='X', clip_on=False,
            zorder=10)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUB1[0:-1]*1e2, color=colors[1], marker='^', clip_on=False,
            zorder=10)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUPOSB1[0:-1]*1e2, color=colors[6], marker='^', clip_on=False,
            zorder=10)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUB2[0:-1]*1e2, color=colors[1], marker='s', clip_on=False,
            zorder=10,markersize=5)
    ax.plot((h[0:-1,0]+h[0:-1,1])/2/1e3, aErrMUPOSB2[0:-1]*1e2, color=colors[6], marker='s', clip_on=False,
            zorder=10,markersize=5)
    ax.plot(-1e3,0,color=colors[6],marker='.',markersize=9,linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='X',linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='^',linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='s',markersize=5,linestyle='')
    ax.plot(-1e3,0,color=colors[1],label='$mu_{M}$')
    ax.plot(-1e3,0,color=colors[6],label='$mu_{M},\textrm{r}_{M}$')

    ax.set_xlim([0,np.max(h[:,0])/1e3])
    ax.set_ylim([1e-2,1e2])
    plt.xlabel('Altitude [km]', fontsize=font)
    plt.ylabel('Mean gravity error [\%]', fontsize=font)
    ax.tick_params(axis='both', labelsize=font)
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid()
    lines = ax.get_lines()
    legend1 = pyplot.legend([lines[i] for i in [10,11,12,13]], ["A1","A2","B1","B2"], fontsize=font-8,
                            loc=0, ncols=2)
    legend2 = pyplot.legend([lines[i] for i in [14,15]],
                            ["$\mu_{M}$","$\mu_{M},\mathbf{r}_{M}$"], fontsize=font-8,loc=3)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.text(7, 1.3e-1, 'Truth dataset', va='center',fontsize=font-4,rotation=-43,color=colors[6])
    ax.text(17.5, 15*1e0, 'Kepler', va='center',fontsize=font-4,rotation=-18,color='k')
    ax.set_title('$n=100$',fontsize=font)

    fig.savefig('Plots/AA/comparisonOnline.pdf', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')

def plotnMascons():
    nM = np.array([100,200,300,400,500,600,700,800,900,1000])
    nRand = 1
    filePath1 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s0m25mm/'
    filePath2 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm/'

    font = 22
    #colors = list(mcolors.TABLEAU_COLORS.values())
    # Viz palette
    colors = ['#ffd700',
              '#ffb14e',
              '#fa8775',
              '#ea5f94',
              '#cd34b5',
              '#9d02d7',
              '#0000ff']

    plt.gcf()
    fig = plt.figure(figsize=(7.3, 6))
    ax = fig.add_subplot(1,1,1)
    errMUA1 = np.zeros(len(nM))
    errMUPOSA1 = np.zeros(len(nM))
    errMUA2 = np.zeros(len(nM))
    errMUPOSA2 = np.zeros(len(nM))
    errMUB1 = np.zeros(len(nM))
    errMUPOSB1 = np.zeros(len(nM))
    errMUB2 = np.zeros(len(nM))
    errMUPOSB2 = np.zeros(len(nM))

    # Loop mascons
    for jj in range(nRand):
        for ii in range(len(nM)):
            fileMUA1 = filePath1 + 'mascon' + str(nM[ii]) + '_MUMSE_octantLR1E-3_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUA1, "rb"))
            errMUA1[ii] = navOutputsM.nav.aErrTotal3D
            fileMUA1 = filePath1 + 'mascon' + str(nM[ii]) + '_MUPOSMSE_octantLR1E-3_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUA1, "rb"))
            errMUPOSA1[ii] = navOutputsM.nav.aErrTotal3D

            fileMUA2 = filePath2 + 'mascon' + str(nM[ii]) + '_MUMSE_octantLR1E-3_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUA2, "rb"))
            errMUA2[ii] = navOutputsM.nav.aErrTotal3D
            fileMUPOSA2 = filePath2 + 'mascon' + str(nM[ii]) + '_MUPOSMSE_octantLR1E-3_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUPOSA2, "rb"))
            errMUPOSA2[ii] = navOutputsM.nav.aErrTotal3D

            fileMUB1 = filePath1 + 'mascon' + str(nM[ii]) + '_MUMSE_octantLR1E-3_eclipse_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUB1, "rb"))
            errMUB1[ii] = navOutputsM.nav.aErrTotal3D
            fileMUPOSB1 = filePath1 + 'mascon' + str(nM[ii]) + '_MUPOSMSE_octantLR1E-3_eclipse_' + 'rand' + str(jj) \
                          + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUPOSB1, "rb"))
            errMUPOSB1[ii] = navOutputsM.nav.aErrTotal3D

            fileMUB2 = filePath2 + 'mascon' + str(nM[ii]) + '_MUMSE_octantLR1E-3_eclipse_' + 'rand' + str(jj) + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUB2, "rb"))
            errMUB2[ii] = navOutputsM.nav.aErrTotal3D
            fileMUPOSB2 = filePath2 + 'mascon' + str(nM[ii]) + '_MUPOSMSE_octantLR1E-3_eclipse_' + 'rand' + str(jj) \
                          + '.pck'
            navParamsM, navOutputsM = pickle.load(open(fileMUPOSB2, "rb"))
            errMUPOSB2[ii] = navOutputsM.nav.aErrTotal3D

        ax.plot(nM, errMUA1*1e2, color=colors[1], marker='.', clip_on=False,markersize=9)
        ax.plot(nM, errMUPOSA1*1e2, color=colors[6], marker='.', clip_on=False,markersize=9)
        ax.plot(nM, errMUA2*1e2, color=colors[1], marker='X', clip_on=False,zorder=10)
        ax.plot(nM, errMUPOSA2*1e2, color=colors[6], marker='X', clip_on=False)
        ax.plot(nM, errMUB1*1e2, color=colors[1], marker='^', clip_on=False,zorder=10)
        ax.plot(nM, errMUPOSB1*1e2, color=colors[6], marker='^', clip_on=False)
        ax.plot(nM, errMUB2*1e2, color=colors[1], marker='s', clip_on=False,zorder=10,markersize=5)
        ax.plot(nM, errMUPOSB2*1e2, color=colors[6], marker='s', clip_on=False,markersize=5)

    ax.plot(-1e3,0,color=colors[6],marker='.',markersize=9,linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='X',linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='^',linestyle='')
    ax.plot(-1e3,0,color=colors[6],marker='s',markersize=5,linestyle='')
    ax.plot(-1e3,0,color=colors[1],label='$mu_{M}$')
    ax.plot(-1e3,0,color=colors[6],label='$mu_{M},\textrm{r}_{M}$')

    ax.set_xlim([nM[0],nM[-1]])
    ax.set_ylim([1e0,6])
    ax.set_xticks([200,400,600,800,1000])
    plt.xlabel('$n$ [-]', fontsize=font)
    plt.ylabel('Global gravity error [\%]', fontsize=font)
    ax.tick_params(axis='both', labelsize=font)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid()

    lines = ax.get_lines()
    legend1 = pyplot.legend([lines[i] for i in [8,9,10,11]], ["A1","A2","B1","B2"], fontsize=font-8, loc=0, ncols=2)
    legend2 = pyplot.legend([lines[i] for i in [12,13]], ["$\mu_{M}$","$\mu_{M},\mathbf{r}_{M}$"], fontsize=font-8,
                            loc=3)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    fig.savefig('Plots/AA/comparisonOnlinenM.pdf', bbox_inches='tight')

def plotGravErr2D():
    fileMascon = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s0m25mm/mascon100_MUPOSMSE_octantLR1E-3_rand0.pck'
    fileMasconEjecta = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s0m25mm/mascon100_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMascon, "rb"))
    navParamsMEjecta, navOutputsMEjecta = pickle.load(open(fileMasconEjecta, "rb"))

    # Plot the global gravity results
    vertList, faceList, nVertex, nFacet = loadPolyFromFileToList(polyFilename)
    xyzPoly = np.array(vertList)
    color_smallbody = [105/255, 105/255, 105/255]

    rTruth = navOutputsM.sim.rTruth_CA_A
    #rEjecta = np.reshape(navOutputsMEjecta.sim.rEjecta_EA_A,(25*10,3))
    #rEjecta = np.reshape(navOutputsMEjecta.sim.rEjecta_EA_A[:,0:5,:], (50,3))
    rEjecta = navOutputsMEjecta.sim.rEjecta_EA_A[0,0:50,0:3]
    Xy2D = navOutputsM.sim.Xy2D
    Yx2D = navOutputsM.sim.Yx2D
    Xz2D = navOutputsM.sim.Xz2D
    Zx2D = navOutputsM.sim.Zx2D
    Yz2D = navOutputsM.sim.Yz2D
    Zy2D = navOutputsM.sim.Zy2D
    accMErrXY2D = navOutputsM.nav.aErrXY2D
    accMErrXZ2D = navOutputsM.nav.aErrXZ2D
    accMErrYZ2D = navOutputsM.nav.aErrYZ2D
    accMEjectaErrXY2D = navOutputsMEjecta.nav.aErrXY2D
    accMEjectaErrXZ2D = navOutputsMEjecta.nav.aErrXZ2D
    accMEjectaErrYZ2D = navOutputsMEjecta.nav.aErrYZ2D
    acc0ErrXY2D = navOutputsM.nav.a0ErrXY2D
    acc0ErrXZ2D = navOutputsM.nav.a0ErrXZ2D
    acc0ErrYZ2D = navOutputsM.nav.a0ErrYZ2D
    acc0ErrYZ2D[0,0] = 0
    maxErr = 0.1
    accMErrXY2D[accMErrXY2D > maxErr] = maxErr
    accMErrXZ2D[accMErrXZ2D > maxErr] = maxErr
    accMErrYZ2D[accMErrYZ2D > maxErr] = maxErr
    acc0ErrXY2D[acc0ErrXY2D > maxErr] = maxErr
    acc0ErrXZ2D[acc0ErrXZ2D > maxErr] = maxErr
    acc0ErrYZ2D[acc0ErrYZ2D > maxErr] = maxErr
    accMEjectaErrXY2D[accMEjectaErrXY2D > maxErr] = maxErr
    accMEjectaErrXZ2D[accMEjectaErrXZ2D > maxErr] = maxErr
    accMEjectaErrYZ2D[accMEjectaErrYZ2D > maxErr] = maxErr
    font = 20
    plt.gcf()
    fig, ax = plt.subplots(2,3, figsize=(12,6), gridspec_kw={'width_ratios': [1, 1, 1]})

    ticks = np.linspace(0, maxErr, 11)*100
    minX = -50*1e3
    maxX = 50*1e3

    colorsGray = ['#FFFFFF',
                  '#D9DDDC',
                  '#676767']

    cs = ax[0,0].contourf(Xy2D/1e3, Yx2D/1e3, accMErrXY2D*100, levels=100, vmax=maxErr*1e2, cmap=get_cmap("viridis"))
    ax[0,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
    ax[0,0].plot(rTruth[:,0]/1e3, rTruth[:,1]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[0,0].set_ylabel('$y$ [km]', fontsize=font)
    ax[0,0].tick_params(axis='both', labelsize=font)
    ax[0,0].set_xlim([minX/1e3, maxX/1e3])
    ax[0,0].set_ylim([minX/1e3, maxX/1e3])
    ax[0,0].set_xticks([-50,-25,0,25,50])
    ax[0,0].set_yticks([-50,-25,0,25,50])
    ax[0,0].set_xticklabels([])

    cs = ax[0,1].contourf(Xz2D/1e3, Zx2D/1e3, accMErrXZ2D*100, levels=100, vmax=maxErr*1e2, cmap=get_cmap("viridis"))
    ax[0,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax[0,1].plot(rTruth[:,0]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[0,1].set_ylabel('$z$ [km]', fontsize=font)
    ax[0,1].tick_params(axis='both', labelsize=font)
    ax[0,1].set_title('On-orbit', fontsize=font)
    ax[0,1].set_xlim([minX/1e3, maxX/1e3])
    ax[0,1].set_ylim([minX/1e3, maxX/1e3])
    ax[0,1].set_xticks([-50,-25,0,25,50])
    ax[0,1].set_xticklabels([])
    ax[0,1].set_yticklabels([])

    cs = ax[0,2].contourf(Yz2D/1e3, Zy2D/1e3, accMErrYZ2D*100, levels=100, vmax=maxErr*1e2, cmap=get_cmap("viridis"))
    ax[0,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax[0,2].plot(rTruth[:,1]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[0,2].set_ylabel('$z$ [km]', fontsize=font)
    ax[0,2].tick_params(axis='both', labelsize=font)
    ax[0,2].set_xlim([minX/1e3, maxX/1e3])
    ax[0,2].set_ylim([minX/1e3, maxX/1e3])
    ax[0,2].set_xticks([-50,-25,0,25,50])
    ax[0,2].set_xticklabels([])
    ax[0,2].set_yticklabels([])

    cs = ax[1,0].contourf(Xy2D/1e3, Yx2D/1e3, accMEjectaErrXY2D*100, levels=100, vmax=maxErr*1e2,
                          cmap=get_cmap("viridis"))
    ax[1,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
    ax[1,0].plot(rTruth[:,0]/1e3, rTruth[:,1]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[1,0].plot(rEjecta[:,0]/1e3, rEjecta[:,1]/1e3, color=colorsGray[1],linestyle='',marker='.',markersize=1.5)
    ax[1,0].set_xlabel('$x$ [km]', fontsize=font)
    ax[1,0].set_ylabel('$y$ [km]', fontsize=font)
    ax[1,0].tick_params(axis='both', labelsize=font)
    ax[1,0].set_xlim([minX/1e3, maxX/1e3])
    ax[1,0].set_ylim([minX/1e3, maxX/1e3])
    ax[1,0].set_xticks([-50,-25,0,25,50])
    ax[1,0].set_yticks([-50,-25,0,25,50])

    cs = ax[1,1].contourf(Xz2D/1e3, Zx2D/1e3, accMEjectaErrXZ2D*100, levels=100, vmax=maxErr*1e2,
                          cmap=get_cmap("viridis"))
    ax[1,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax[1,1].plot(rTruth[:,0]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[1,1].plot(rEjecta[:,0]/1e3, rEjecta[:,2]/1e3, color=colorsGray[1],linestyle='',marker='.',markersize=1.5)
    ax[1,1].set_xlabel('$x$ [km]', fontsize=font)
    ax[1,1].set_ylabel('$z$ [km]', fontsize=font)
    ax[1,1].tick_params(axis='both', labelsize=font)
    ax[1,1].set_title('On-orbit with ejecta', fontsize=font)
    ax[1,1].set_xlim([minX/1e3, maxX/1e3])
    ax[1,1].set_ylim([minX/1e3, maxX/1e3])
    ax[1,1].set_xticks([-50,-25,0,25,50])
    ax[1,1].set_yticklabels([])

    cs = ax[1,2].contourf(Yz2D/1e3, Zy2D/1e3, accMEjectaErrYZ2D*100, levels=100, vmax=maxErr*1e2,
                          cmap=get_cmap("viridis"), extend='max')
    ax[1,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax[1,2].plot(rTruth[:,1]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax[1,2].plot(rEjecta[:,1]/1e3, rEjecta[:,2]/1e3, color=colorsGray[1],linestyle='',marker='.',markersize=1.5)
    ax[1,2].set_xlabel('$y$ [km]', fontsize=font)
    ax[1,2].set_ylabel('$z$ [km]', fontsize=font)
    ax[1,2].tick_params(axis='both', labelsize=font)
    ax[1,2].set_xlim([minX/1e3, maxX/1e3])
    ax[1,2].set_ylim([minX/1e3, maxX/1e3])
    ax[1,2].set_xticks([-50,-25,0,25,50])
    ax[1,2].set_yticklabels([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(cs, cax=cbar_ax, shrink=0.95, extend='max', ticks=ticks)
    cbar.set_label('Gravity error [\%]', fontsize=font, rotation=90)
    cbar.ax.set_yticklabels(['$0$', '', '$2$', '', '$4$', '', '$6$', '', '$8$', '', '$10$'])
    cbar.ax.tick_params(labelsize=font)

    plt.savefig('Plots/AA/gravity2Dejecta.pdf', format='pdf')

def plotMascon():
    color_smallbody = [105/255, 105/255, 105/255]
    font = 22

    fileMascon100 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    fileMascon1000 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm/mascon1000_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'

    navParams, navOutputs = pickle.load(open(fileMascon100,"rb"))
    posM100 = navParams.gravityEst.posM
    posM100init = navParams.gravityEst.posM0
    muM100 = navParams.gravityEst.muM
    navParams, navOutputs = pickle.load(open(fileMascon1000,"rb"))
    posM1000init = navParams.gravityEst.posM0
    posM1000 = navParams.gravityEst.posM
    muM1000 = navParams.gravityEst.muM
    xyzVertex = navParams.smallbody.xyzVertex
    orderFacet = navParams.smallbody.orderFacet - 1

    # Plot the relative 3D position
    plt.gcf()
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot_trisurf(xyzVertex[:, 0]/1e3, xyzVertex[:,1]/1e3, xyzVertex[:,2]/1e3, triangles=orderFacet,
                     color=color_smallbody, zorder=0, alpha=0.1)
    p1 = ax1.scatter3D(posM100[:,0]/1e3, posM100[:,1]/1e3, posM100[:,2]/1e3, c=muM100, cmap=get_cmap("viridis"),
                       norm=mpl.colors.LogNorm(vmin=1,vmax=np.max(muM100)))
    mean = np.sum(muM100/101)
    perc0 = len(muM100[muM100<1])/101*100
    max = np.max(muM100)
    meanpos = np.sum(np.linalg.norm(posM100-posM100init,axis=1))/101
    print([mean,max,perc0,meanpos])
    ax1.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax1.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    ax1.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax1.set_xlim(-20,20)
    ax1.set_ylim(-20,20)
    ax1.set_zlim(-20,20)
    ax1.set_xticks([-10,0,10])
    ax1.set_yticks([-10,0,10])
    ax1.set_zticks([-10,0,10])
    ax1.tick_params(axis='both', labelsize=font)
    ax1.set_facecolor('white')
    ax1.set_title('$n=100$', fontsize=font)

    set_axes_equal(ax1)
    ax1.view_init(elev=59., azim=360 - 144)

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.plot_trisurf(xyzVertex[:, 0]/1e3, xyzVertex[:,1]/1e3, xyzVertex[:,2]/1e3, triangles=orderFacet,
                     color=color_smallbody, zorder=0, alpha=0.1)
    #muM1000[muM1000<1] = 1
    p2 = ax2.scatter3D(posM1000[:,0]/1e3, posM1000[:,1]/1e3, posM1000[:,2]/1e3, c=muM1000, cmap=get_cmap("viridis"),
                       norm=mpl.colors.LogNorm(vmin=1,vmax=np.max(muM100)))
    mean = np.sum(muM1000 / 1001)
    perc0 = len(muM1000[muM1000 < 1]) / 1001 * 100
    max = np.max(muM1000)
    meanpos = np.sum(np.linalg.norm(posM1000-posM1000init,axis=1))/1001
    print([mean,max,perc0,meanpos])
    ax2.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax2.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    #ax2.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax2.set_xlim(-20,20)
    ax2.set_ylim(-20,20)
    ax2.set_zlim(-20,20)
    ax2.set_xticks([-10,0,10])
    ax2.set_yticks([-10,0,10])
    ax2.set_zticks([-10,0,10])
    ax2.tick_params(axis='both', labelsize=font)
    ax2.set_facecolor('white')
    ax2.set_title('$n=1000$', fontsize=font)
    set_axes_equal(ax2)
    ax2.view_init(elev=59., azim=360 - 144)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.25, 0.01, 0.45])
    cbar = fig.colorbar(p2, cax=cbar_ax, shrink=0.7, extend='min', ticks=[1,10,100,1000,10000])
    cbar.ax.tick_params(labelsize=font)
    cbar.ax.yaxis.get_offset_text().set_fontsize(font)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.ax.set_yticklabels(['$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$'])
    cbar.set_label('$\mu_M$ [m$^3$/s$^2$]', rotation=90, fontsize=font)
    plt.savefig('Plots/AA/mascon.pdf', format='pdf')

def plotMascon0():
    ####################################################################################################################
    x = np.linspace(-17,17,16)
    y = np.linspace(-17,17,16)
    z = np.linspace(-17,17,16)
    Xxy, Yxy = np.meshgrid(x,y)
    Xxz, Zxz = np.meshgrid(x,z)
    Yyz, Zyz = np.meshgrid(y,z)
    Zxy = 0*Xxy + 0*Yxy
    Yxz = 0*Xxz + 0*Zxz
    Xyz = 0*Yyz + 0*Zyz

    # Load polyhedron
    vertList, faceList, nVertex, nFacet = loadPolyFromFileToList(polyFilename)
    xyzPoly = np.array(vertList)
    idxPoly = np.array(faceList)

    # Get polyhedron and landmarks
    color_smallbody = [105/255, 105/255, 105/255]

    # Plot the relative 3D position
    filenameMascons = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm/mascon100_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    posM0 = navParamsM.gravityEst.posM0
    Nsector = int(len(posM0)/8)

    cmap = mpl.cm.get_cmap("viridis")
    bottom = cmap(0)
    top = cmap(cmap.N)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.set_aspect('equal')
    ax.plot_surface(Xxy, Yxy, Zxy, zorder=1, alpha=0.05, color='blue')
    ax.plot_surface(Xxz, Yxz, Zxz, zorder=1, alpha=0.05, color='blue')
    ax.plot_surface(Xyz, Yyz, Zyz, zorder=1, alpha=0.05, color='blue')
    ax.plot_trisurf(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, triangles=idxPoly-1, color=color_smallbody, zorder=10, alpha=.4)
    for ii in range(8):
        if ii == 0:
            ax.plot(posM0[ii*Nsector+1:(ii+1)*Nsector+1,0]/1e3, posM0[ii*Nsector+1:(ii+1)*Nsector+1,1]/1e3,
                     posM0[ii*Nsector+1:(ii+1)*Nsector+1,2]/1e3, color=bottom, linestyle='', marker='.', markersize=7.5, zorder=20,label='$\mu_{M_k}=0$')
        else:
            ax.plot(posM0[ii*Nsector+1:(ii+1)*Nsector+1,0]/1e3, posM0[ii*Nsector+1:(ii+1)*Nsector+1,1]/1e3,
                     posM0[ii*Nsector+1:(ii+1)*Nsector+1,2]/1e3, color=bottom, linestyle='', marker='.', markersize=7.5, zorder=20)
    ax.plot(0,0,0,color=top,linestyle='',marker='.',markersize=10, zorder=20,label='$\mu_{M_n}=\mu$')
    #ax.set_xlabel('$x$ [km]', fontsize=14)
    #ax.set_ylabel('$y$ [km]', fontsize=14)
    #ax.set_zlabel('$z$ [km]', fontsize=14)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    ax.tick_params(axis='both', labelsize=14)
    ax.set_facecolor('white')

    set_axes_equal(ax)
    ax.view_init(elev=59., azim=360-144)
    #ax.text(0,0,0, '$\mu$', va='center',fontsize=font-4,rotation=0,color=top)

    plt.tight_layout()
    plt.savefig('Plots/AA/masconInit.pdf', format='pdf', bbox_inches='tight')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def processResults():
    filePath1 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s0m25mm'
    filePath2 = 'Results/eros_polylsqUKF_10orbits_a34kmi45deg_60s5m25mm'
    nM = 100

    errTotal = np.zeros((4,4))
    fileMasconA1 = filePath1 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconA1, "rb"))
    errTotal[0,1] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    errTotal[0,3] = navOutputsM.nav.aErrTotal3D
    fileMasconA1 = filePath1 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconA1, "rb"))
    errTotal[0,0] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    errTotal[0,2] = navOutputsM.nav.aErrTotal3D

    fileMasconA2 = filePath2 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconA2, "rb"))
    errTotal[1,1] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaA2 = filePath2 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA2, "rb"))
    errTotal[1,3] = navOutputsM.nav.aErrTotal3D
    fileMasconA2 = filePath2 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconA2, "rb"))
    errTotal[1,0] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaA2 = filePath2 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA2, "rb"))
    errTotal[1,2] = navOutputsM.nav.aErrTotal3D

    fileMasconB1 = filePath1 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconB1, "rb"))
    errTotal[2,1] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaB1 = filePath1 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_eclipseejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaB1, "rb"))
    errTotal[2,3] = navOutputsM.nav.aErrTotal3D
    fileMasconB1 = filePath1 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconB1, "rb"))
    errTotal[2,0] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaB1 = filePath1 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_eclipseejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaB1, "rb"))
    errTotal[2,2] = navOutputsM.nav.aErrTotal3D

    fileMasconB2 = filePath2 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconB2, "rb"))
    errTotal[3,1] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaB2 = filePath2 + '/mascon' + str(int(nM)) + '_MUPOSMSE_octantLR1E-3_eclipseejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaB2, "rb"))
    errTotal[3,3] = navOutputsM.nav.aErrTotal3D
    fileMasconB2 = filePath2 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_eclipse_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconB2, "rb"))
    errTotal[3,0] = navOutputsM.nav.aErrTotal3D
    fileMasconEjectaB2 = filePath2 + '/mascon' + str(int(nM)) + '_MUMSE_octantLR1E-3_eclipseejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaB2, "rb"))
    errTotal[3,2] = navOutputsM.nav.aErrTotal3D

    print(errTotal*1e2)

    nM = np.array([100,500,1000])
    tcpu = np.zeros((6,4))
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[0])) + '_MUMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[0,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[1])) + '_MUMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[1,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[2])) + '_MUMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[2,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[0])) + '_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[3,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[1])) + '_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[4,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    fileMasconEjectaA1 = filePath1 + '/mascon' + str(int(nM[2])) + '_MUPOSMSE_octantLR1E-3ejectaM50_rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(fileMasconEjectaA1, "rb"))
    tcpu[5,0:4] = np.array([np.sum(navOutputsM.nav.tcpuDMCUKF)/len(navOutputsM.nav.tcpuDMCUKF),
                            np.max(navOutputsM.nav.tcpuDMCUKF),
                            np.sum(navOutputsM.nav.tcpuGravEst)/len(navOutputsM.nav.tcpuGravEst),
                            np.max(navOutputsM.nav.tcpuGravEst)])
    print(tcpu)


if __name__ == "__main__":
    polyFilename = '/Users/julio/basilisk/supportData/LocalGravData/eros007790.tab'
    font = 22
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Plots
    plotMascon0()
    plotOnline()
    plotnMascons()
    plotDMCUKF()
    plotMascon()
    plotGravErr2D()

    processResults()
    plt.show()


