import pickle

from Plotting.plots_dmcukf import *
from Plotting.generalPlots import *

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from PIL import Image

def plotPropRMS():
    plt.gcf()
    fig5 = plt.figure(figsize=(7.3,6))
    ax5 = fig5.add_subplot(1,1,1)
    #filenameProp = 'propData_r34km_sqr.pck' # 20 km
    filePath = 'Results/Propagation/medOrbit'

    #percM = np.zeros(4)
    #minRMS_M = np.zeros(4)
    #minpercM = np.zeros(4)
    #for ii in range(4):
    #    idx = int((ii+1)*len(time)/4)
    #    minRMS_lsqSH = np.min(RMS_intSH[idx-1,:])
    #    percM[ii] = np.sum(RMS_M[idx-1,:] < minRMS_lsqSH)/1000*100
    #    minRMS_M[ii] = np.min(RMS_M[idx-1,:])
    #    minpercM[ii] = (minRMS_M[ii]/minRMS_lsqSH)*100
    #print(percM)
    #print(minRMS_M)
    #print(minpercM)

    #nRand = len(RMS_M)
    #_, nRand = RMS_M.shape
    nRand = 562

    # Compute estimation
    #ax5.plot(time/(3600*24), RMS0, color='black', zorder=10,linestyle='dashdot')
    for ii in range(7):
        filenameProp = filePath + '/SH' + str(ii+2) + 'th.pck'
        time, _, RMS = pickle.load(open(filenameProp, "rb"))
        ax5.plot(time/(3600*24), RMS, color=colors[ii], zorder=10, linestyle='dashed')
    #ax5.plot(-1e3, 0, color=colors[0], linewidth=0.5)
    #RMS_MMean = np.sum(RMS_M,axis=1)/1000
    #ax5.plot(time / (3600 * 24), RMS_MMean, color='blue', linewidth=1,zorder=10)
    for ii in range(nRand):
        filenameProp = filePath + '/M80/rand' + str(ii) + '.pck'
        time, _, RMS = pickle.load(open(filenameProp, "rb"))
        ax5.plot(time/(3600*24), RMS, color=colors[0], linewidth=0.05)
        if ii == 0:
            RMSMean = RMS
        else:
            RMSMean += RMS
    RMSMean /= nRand
    ax5.plot(time / (3600*24), RMSMean, color=colors[0], linewidth=2)
    ax5.set_xlim([time[0]/(3600*24),time[-1]/(3600*24)])
    ax5.set_ylim([0,2000])
    #ax5.set_ylim([1e-1,1e4])
    plt.xlabel('Time [days]', fontsize=font)
    plt.ylabel('Propagation RMSE [m]', fontsize=font)
    ax5.tick_params(axis='both', labelsize=font)
    #ax5.set_yscale('log')
    #ax.set_yticks([0.1,1,10,1e2,1e3,1e4,1e5,1e6,1e7])
    ax5.xaxis.set_minor_locator(AutoMinorLocator())
    ax5.yaxis.set_minor_locator(AutoMinorLocator())
    #y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1, numticks=10)
    #ax5.yaxis.set_minor_locator(y_minor)
    #ax5.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax5.grid()
    lines = ax5.get_lines()
    legend1 = pyplot.legend([lines[i] for i in [0]],
                            ["Keplerian"], loc=2, fontsize=font-4, bbox_to_anchor=(1.02, 1),borderaxespad=0.)
    legend2 = pyplot.legend([lines[i] for i in [1,2,3,4,5,6,7]],
                            ["SH deg=2", "SH deg=3","SH deg=4","SH deg=5","SH deg=6","SH deg=7","SH deg=8"], loc=2,
                            fontsize=font-4, bbox_to_anchor=(1.02, 0.887),borderaxespad=0.)
    legend3 = pyplot.legend([lines[i] for i in [8]],
                            ["M $n$=641"], loc=2, fontsize=font-4, bbox_to_anchor=(1.02, 0.3279),borderaxespad=0.)
    ax5.add_artist(legend3)
    ax5.add_artist(legend2)
    ax5.add_artist(legend1)

    plt.tight_layout()

    plt.savefig('Plots/AA/propRMS.pdf', bbox_extra_artists=(legend1, legend2, legend3), bbox_inches='tight')

def plotDMCUKF():
    font = 22
    """Plot the position estimation error and associated covariance."""
    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1Winit/rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    time = navOutputsM.sim.t
    r_err = navOutputsM.nav.rEst_CA_A - navOutputsM.sim.rTruth_CA_A
    Ppos_A = navOutputsM.nav.Ppos_A
    P = navOutputsM.nav.P
    navCamSolutionRaw = navOutputsM.sim.navCamSolution

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
    fig7, ax7 = plt.subplots(3, sharex=True, figsize=(12,10))
    fig7.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax7[0].plot(time/(3600*24), r_err[:,0]*navCamSolution, color='blue', marker='.',markersize=3,linestyle='')
    ax7[0].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,0,0])*navCamSolution, 'k',linewidth=1)
    ax7[0].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,0,0])*navCamSolution, 'k',linewidth=1)

    ax7[1].plot(time/(3600*24), r_err[:,1]*navCamSolution, color='blue', marker='.',markersize=3,linestyle='')
    ax7[1].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,1,1])*navCamSolution, 'k', linewidth=1)
    ax7[1].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,1,1])*navCamSolution, 'k',linewidth=1)

    ax7[2].plot(time/(3600*24), r_err[:,2]*navCamSolution, color='blue', marker='.',markersize=3,linestyle='')
    ax7[2].plot(time/(3600*24), 3*np.sqrt(Ppos_A[:,2,2])*navCamSolution, 'k',linewidth=1)
    ax7[2].plot(time/(3600*24), -3*np.sqrt(Ppos_A[:,2,2])*navCamSolution, 'k',linewidth=1)

    ax7[1].plot(-1e3, 0, color='blue',linewidth=1,marker='.', markersize=6, linestyle='',label='UKF')
    ax7[1].plot(-1e3, 0, color='black', linewidth=1, label='3-$\sigma$')

    for ii in range(len(tfill)):
        if ii == 0:
            ax7[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1, label='Gaps')
        else:
            ax7[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1)
        ax7[0].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color='red', alpha=.1)
        ax7[2].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color='red', alpha=.1)

    ax7[2].set_xlabel('Time [days]', fontsize=font, labelpad=10)

    ax7[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax7[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax7[2].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])

    ax7[0].set_ylim([-30,30])
    ax7[1].set_ylim([-30,30])
    ax7[2].set_ylim([-30,30])

    ax7[0].set_ylabel('$\delta x$ [m]', fontsize=font)
    ax7[1].set_ylabel('$\delta y$ [m]', fontsize=font)
    ax7[2].set_ylabel('$\delta z$ [m]', fontsize=font)

    ax7[0].tick_params(axis='both', labelsize=font)
    ax7[1].tick_params(axis='both', labelsize=font)
    ax7[2].tick_params(axis='both', labelsize=font)

    ax7[0].set_yticks([-30,-15,0,15,30])
    ax7[1].set_yticks([-30,-15,0,15,30])
    ax7[2].set_yticks([-30,-15,0,15,30])

    ax7[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax7[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax7[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax7[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax7[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax7[2].yaxis.set_minor_locator(AutoMinorLocator())

    ax7[0].set_title('Mascon $n$=641',fontsize=font)

    ax7[0].grid()
    ax7[1].grid()
    ax7[2].grid()

    ax7[1].legend(fontsize=font-4, bbox_to_anchor=(1.01,0.73),borderaxespad=0.)

    #lines = ax[0].get_lines()
    #legend = pyplot.legend([lines[i] for i in [0,1]],
    #                        ["Error","3-$\sigma$ bounds"], loc=2, fontsize=font, bbox_to_anchor=(1.038,1),borderaxespad=0.)
    #ax[0].add_artist(legend)

    plt.tight_layout()
    #fig.savefig('samplefigure', bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
    plt.savefig('Plots/AA/posError.pdf', format='pdf')

    aTruth = navOutputsM.sim.aTruth_A
    aEst = navOutputsM.nav.aTrain
    Pacc_A = navOutputsM.nav.Pacc_A
    for ii in range(len(aEst)):
        aEst[ii, 0:3] += navParamsM.smallbody.mu * np.array(navOutputsM.sim.rTruth_CA_A[ii, 0:3]) / np.linalg.norm(
            navOutputsM.sim.rTruth_CA_A[ii, 0:3])**3
    plt.gcf()
    fig8, ax8 = plt.subplots(3, sharex=True, figsize=(12,10))
    fig8.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax8[0].plot(time/(3600*24), aTruth[:,0]*1e6*navCamSolution, color='blue',linewidth=0.85,zorder=10)
    ax8[1].plot(time/(3600*24), aTruth[:,1]*1e6*navCamSolution, color='blue', linewidth=0.85,zorder=10)
    ax8[2].plot(time/(3600*24), aTruth[:,2]*1e6*navCamSolution, color='blue',linewidth=0.85,zorder=10)

    ax8[0].plot(time/(3600*24), aEst[:,0]*1e6*navCamSolution, color='orange', marker='.', markersize=3, linestyle='')
    ax8[1].plot(time/(3600*24), aEst[:,1]*1e6*navCamSolution, color='orange', marker='.', markersize=3, linestyle='')
    ax8[2].plot(time/(3600*24), aEst[:,2]*1e6*navCamSolution, color='orange', marker='.', markersize=3, linestyle='')

    ax8[0].plot(time/(3600*24), aEst[:,0]*1e6*navCamSolution+3*np.sqrt(Pacc_A[:,0,0])*1e6*navCamSolution, 'k',linewidth=0.75)
    ax8[1].plot(time/(3600*24), aEst[:,1]*1e6*navCamSolution+3*np.sqrt(Pacc_A[:,1,1])*1e6*navCamSolution, 'k',linewidth=0.75)
    ax8[2].plot(time/(3600*24), aEst[:,2]*1e6*navCamSolution+3*np.sqrt(Pacc_A[:,2,2])*1e6*navCamSolution, 'k',linewidth=0.75)
    ax8[0].plot(time/(3600*24), aEst[:,0]*1e6*navCamSolution-3*np.sqrt(Pacc_A[:,0,0])*1e6*navCamSolution, 'k',linewidth=0.75)
    ax8[1].plot(time/(3600*24), aEst[:,1]*1e6*navCamSolution-3*np.sqrt(Pacc_A[:,1,1])*1e6*navCamSolution, 'k',linewidth=0.75)
    ax8[2].plot(time/(3600*24), aEst[:,2]*1e6*navCamSolution-3*np.sqrt(Pacc_A[:,2,2])*1e6*navCamSolution, 'k',linewidth=0.75)

    ax8[1].plot(-1e3, 0, color='blue',linewidth=1,label='Truth')
    ax8[1].plot(-1e3, 0, color='orange', marker='.', markersize=6, linestyle='',label='UKF')
    ax8[1].plot(-1e3, 0, color='black', linewidth=1, label='3-$\sigma$')
    for ii in range(len(tfill)):
        if ii == 0:
            ax8[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1, label='Gaps')
        else:
            ax8[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1)
        ax8[0].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                           alpha=.1)
        ax8[2].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                           alpha=.1)

    plt.xlabel('Time [days]', fontsize=font, labelpad=10)

    ax8[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax8[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax8[2].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])

    ax8[0].set_ylim([-100,100])
    ax8[1].set_ylim([-100,100])
    ax8[2].set_ylim([-100,100])

    ax8[0].set_ylabel('$a_{x}$ [$\mu$m/s$^2$]', fontsize=font)
    ax8[1].set_ylabel('$a_{y}$ [$\mu$m/s$^2$]', fontsize=font)
    ax8[2].set_ylabel('$a_{z}$ [$\mu$m/s$^2$]', fontsize=font)

    ax8[0].tick_params(axis='both', labelsize=font)
    ax8[1].tick_params(axis='both', labelsize=font)
    ax8[2].tick_params(axis='both', labelsize=font)

    ax8[0].set_yticks([-100,-50,0,50,100])
    ax8[1].set_yticks([-100,-50,0,50,100])
    ax8[2].set_yticks([-100,-50,0,50,100])

    ax8[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax8[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax8[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax8[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax8[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax8[2].yaxis.set_minor_locator(AutoMinorLocator())

    ax8[0].set_title('Mascon $n$=641',fontsize=font)

    ax8[0].grid()
    ax8[1].grid()
    ax8[2].grid()

    ax8[1].legend(fontsize=font-4, bbox_to_anchor=(1.01, 0.81), borderaxespad=0.)
    plt.tight_layout()

    plt.savefig('Plots/AA/accError.pdf', format='pdf')

    rErrPerc = np.linalg.norm(navOutputsM.nav.rEst_CA_N - navOutputsM.sim.rTruth_CA_N, axis=1) \
               / np.linalg.norm(navOutputsM.sim.rTruth_CA_N, axis=1)
    rCamErrPerc = np.linalg.norm(navOutputsM.sim.rCamMeas_CA_A - navOutputsM.sim.rTruth_CA_A, axis=1) \
               / np.linalg.norm(navOutputsM.sim.rTruth_CA_N, axis=1)
    aErrPerc = np.linalg.norm(aEst - aTruth, axis=1) / np.linalg.norm(aTruth, axis=1)

    #sumidx = 0
    Ndiv = 10
    tInterval = np.zeros((int(len(time)/Ndiv),2))
    rErrInterval = np.zeros(int(len(time)/Ndiv))
    aErrInterval = np.zeros(int(len(time)/Ndiv))
    navCamInterval = np.zeros(int(len(time)/Ndiv))
    tInterval[:,0] = np.linspace(time[0],time[-1],len(tInterval[:,0])) + Ndiv*30
    tInterval[:,1] = np.linspace(time[0],time[-1],len(tInterval[:,0])) + Ndiv*90
    for ii in range(len(tInterval[:,0])):
        idx = np.where(np.logical_and(time >= tInterval[ii,0], time < tInterval[ii,1]))[0]
        rErrInterval[ii] = np.sum(rErrPerc[idx]) / len(idx)
        aErrInterval[ii] = np.sum(aErrPerc[idx]) / len(idx)
        if np.count_nonzero(~np.isnan(navCamSolution[idx])) > 5:
            navCamInterval[ii] = np.nansum(navCamSolution[idx])/ np.count_nonzero(~np.isnan(navCamSolution[idx]))
        else:
            navCamInterval[ii] = np.nan
    plt.gcf()
    fig9, ax9 = plt.subplots(2, sharex=True, figsize=(12,6.66))
    fig9.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    #ax9[0].plot(time/(3600*24), rCamErrPerc*1e2*navCamSolution, color='black',linestyle='',marker='.',markersize=2.5)
    ax9[0].plot(time/(3600*24), rErrPerc*1e2*navCamSolution, color='orange',linestyle='',marker='.',markersize=2.5)
    ax9[1].plot(time/(3600*24), aErrPerc*1e2*navCamSolution, color='orange', linestyle='',marker='.',markersize=2.5)
    ax9[0].plot((tInterval[:,0]+Ndiv*60)/(3600*24), rErrInterval*1e2*navCamInterval, color='black',linewidth=0.75)
    ax9[1].plot((tInterval[:,0]+Ndiv*60)/(3600*24), aErrInterval*1e2*navCamInterval, color='black',linewidth=0.75)
    ax9[1].plot(-1e3, 0, color='orange',linewidth=1,marker='.', markersize=6, linestyle='',label='UKF')
    ax9[1].plot(-1e3, 0, color='black', linewidth=1, label='Mean')
    for ii in range(len(tfill)):
        if ii == 0:
            ax9[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1, label='Gaps')
        else:
            ax9[1].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                               alpha=.1)
        ax9[0].fill_between(tfill[ii, 0:2] / (3600 * 24), yfillLow[ii, 0:2], yfillSup[ii, 0:2], color='red',
                           alpha=.1)

    plt.xlabel('Time [days]', fontsize=font, labelpad=10)

    ax9[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax9[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax9[0].set_ylim([1e-4,1e0])
    ax9[1].set_ylim([1e-1,1000])
    ax9[0].set_yscale('log')
    ax9[1].set_yscale('log')

    ax9[0].set_ylabel('$\delta r^{\mathrm{data}}$ [\%]', fontsize=font)
    ax9[1].set_ylabel('$\delta a^{\mathrm{data}}$ [\%]', fontsize=font)

    ax9[0].tick_params(axis='both', labelsize=font)
    ax9[1].tick_params(axis='both', labelsize=font)
    ax9[0].set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])

    ax9[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax9[1].xaxis.set_minor_locator(AutoMinorLocator())
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1, numticks=10)
    ax9[0].yaxis.set_minor_locator(y_minor)
    ax9[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    #ax8[0].set_yticks([-100,-50,0,50,100])
    #ax8[1].set_yticks([-100,-50,0,50,100])

    ax9[0].set_title('Mascon $n$=641',fontsize=font)

    ax9[0].grid()
    ax9[1].grid()
    handles, labels = ax9[1].get_legend_handles_labels()
    fig9.legend(handles, labels,fontsize=font-4,bbox_to_anchor=(1.11, 0.65), borderaxespad=0.) #loc='upper center')
    plt.tight_layout()

    plt.savefig('Plots/AA/dataError.pdf', bbox_inches='tight')


def plotNavCamErr():
    """Plot the position estimation error and associated covariance."""
    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1Winit/rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    time = navOutputsM.sim.t
    r_err = navOutputsM.sim.rCamMeas_CA_A - navOutputsM.sim.rTruth_CA_A
    P = navOutputsM.sim.RCamMeas
    navCamSolutionRaw = navOutputsM.sim.navCamSolution
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
            yfillSupRaw.append([100,100])
            yfillLowRaw.append([-100,-100])
    tfill = np.array(tfillRaw)
    yfillSup = np.array(yfillSupRaw)
    yfillLow = np.array(yfillLowRaw)

    """Plot the position estimation error and associated covariance."""
    font = 22
    plt.gcf()
    fig4, ax4 = plt.subplots(3, sharex=True, figsize=(12,10))
    fig4.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax4[0].plot(time/(3600*24), r_err[:,0]*navCamSolution, 'b', label='Error', marker='.',markersize=3,linestyle='')
    ax4[0].plot(time/(3600*24), 3*np.sqrt(P[:,0])*navCamSolution, 'k',linewidth=0.75)
    ax4[0].plot(time/(3600*24), -3*np.sqrt(P[:,0])*navCamSolution, 'k',linewidth=0.75)

    ax4[1].plot(-1e3, 0, 'b', marker='.',markersize=5.5,linestyle='', label='Error')
    ax4[1].plot(time/(3600*24), 3*np.sqrt(P[:,4])*navCamSolution, 'k', label=r'3-$\sigma$',linewidth=0.75)
    ax4[1].plot(time/(3600*24), -3*np.sqrt(P[:,4])*navCamSolution, 'k',linewidth=0.75)
    ax4[1].plot(time/(3600*24), r_err[:,1]*navCamSolution, 'b', marker='.',markersize=3,linestyle='')

    ax4[2].plot(time/(3600*24), r_err[:,2]*navCamSolution, 'b', marker='.',markersize=3,linestyle='')
    ax4[2].plot(time/(3600*24), 3*np.sqrt(P[:,8])*navCamSolution, 'k',linewidth=0.75)
    ax4[2].plot(time/(3600*24), -3*np.sqrt(P[:,8])*navCamSolution, 'k',linewidth=0.75)

    for ii in range(len(tfill)):
        if ii == 0:
            ax4[1].fill_between(tfill[ii,0:2] / (3600*24), yfillLow[ii,0:2], yfillSup[ii,0:2], color='red',
                               alpha=.1, label='Gaps')
        else:
            ax4[1].fill_between(tfill[ii,0:2] / (3600*24), yfillLow[ii,0:2], yfillSup[ii,0:2], color='red',
                               alpha=.1)
        ax4[0].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color='red', alpha=.1)
        ax4[2].fill_between(tfill[ii,0:2]/(3600*24),yfillLow[ii,0:2],yfillSup[ii,0:2],color='red', alpha=.1)

    plt.xlabel('Time [days]', fontsize=font, labelpad=10)

    ax4[0].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax4[1].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])
    ax4[2].set_xlim([time[0]/(24*3600), time[-1]/(24*3600)])

    ax4[0].set_ylim([-30,30])
    ax4[1].set_ylim([-30,30])
    ax4[2].set_ylim([-30,30])

    ax4[0].set_ylabel('$\delta x$ [m]', fontsize=font)
    ax4[1].set_ylabel('$\delta y$ [m]', fontsize=font)
    ax4[2].set_ylabel('$\delta z$ [m]', fontsize=font)

    ax4[0].tick_params(axis='both', labelsize=font)
    ax4[1].tick_params(axis='both', labelsize=font)
    ax4[2].tick_params(axis='both', labelsize=font)

    ax4[0].set_yticks([-30,-15,0,15,30])
    ax4[1].set_yticks([-30,-15,0,15,30])
    ax4[2].set_yticks([-30,-15,0,15,30])

    ax4[0].grid()
    ax4[1].grid()
    ax4[2].grid()
    ax4[1].legend(fontsize=font-4, bbox_to_anchor=(1.01,0.73),borderaxespad=0.)
    #ax4[0].set_title('Mascon $n$=400',fontsize=font)
    plt.tight_layout()

    plt.savefig('Plots/AA/navCamError.pdf', format='pdf')



def plotGravErrMean():
    font = 22
    colors = list(mcolors.TABLEAU_COLORS.values())
    #colors=['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
    # #https: // stats.stackexchange.com / questions / 118033 / best - series - of - colors - to - use -
    #for -differentiating - series - in -publication - quality
    nRand = 1000

    plt.gcf()
    fig = plt.figure(figsize=(7.3,6))
    ax = fig.add_subplot(1,1,1)

    # Declare files
    filenameSpherharmInt = ['Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm2th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm3th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm4th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm5th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm6th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm7th.pck',
                         'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm8th.pck']
    filenameSpherharm = ['Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm2th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm3th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm4th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm5th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm6th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm7th.pck',
                         'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm8th.pck']
    filenameMascons = ['Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons80sqrLAM1E-1WinitV2/',
                       'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/',
                       'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/',
                       'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/']
    for jj in range(len(filenameSpherharm)):
        aErrXYZ3DInterval = np.zeros((nRand, 39))
        filename_jj = filenameSpherharm[jj]
        navParams, navOutputs = pickle.load(open(filename_jj, "rb"))
        rXYZ3DInterval = navOutputs.nav.rXYZ3DInterval
        plt.plot(rXYZ3DInterval[:,1]/1e3,navOutputs.nav.aErrXYZ3DInterval*1e2,color=colors[jj],linestyle='dashed',marker='s',markersize=3)
    #for jj in range(len(filenameSpherharmInt)):
    #    aErrXYZ3DInterval = np.zeros((nRand, 39))
    #    filename_jj = filenameSpherharmInt[jj]
    #    navParams, navOutputs = pickle.load(open(filename_jj, "rb"))
    #    rXYZ3DInterval = navOutputs.nav.rXYZ3DInterval
    #    plt.plot(rXYZ3DInterval[:,1]/1e3,navOutputs.nav.aErrXYZ3DInterval*1e2,color=colors[jj],linestyle='dotted')
    plt.plot(rXYZ3DInterval[:,1]/1e3,navOutputs.nav.a0ErrXYZ3DInterval*1e2, color='black',linewidth=1.5,linestyle='dashdot')
    for jj in range(len(filenameMascons)):
        aErrXYZ3DInterval = np.zeros((nRand, 39))
        for ii in range(nRand):
            filename_ii = filenameMascons[jj] + 'rand' + str(ii) + '.pck'
            navParams, navOutputs = pickle.load(open(filename_ii, "rb"))
            aErrXYZ3DInterval[ii,:] = navOutputs.nav.aErrXYZ3DInterval
        rXYZ3DInterval = navOutputs.nav.rXYZ3DInterval
        aErrXYZ3DIntervalMin = np.min(aErrXYZ3DInterval, axis=0)
        aErrXYZ3DIntervalMax = np.max(aErrXYZ3DInterval, axis=0)
        aErrXYZ3DIntervalMean = np.sum(aErrXYZ3DInterval, axis=0) / nRand
        ax.fill_between(rXYZ3DInterval[:,1]/1e3, aErrXYZ3DIntervalMin*1e2, aErrXYZ3DIntervalMax*1e2, color=colors[3-jj],
                        alpha=.15)
        plt.plot(rXYZ3DInterval[:,1]/1e3,aErrXYZ3DIntervalMean*1e2,color=colors[3-jj],marker='.')
        plt.plot(rXYZ3DInterval[:,1]/1e3,aErrXYZ3DIntervalMin*1e2,color=colors[3-jj],linewidth=0.3)
        plt.plot(rXYZ3DInterval[:,1]/1e3,aErrXYZ3DIntervalMax*1e2,color=colors[3-jj],linewidth=0.3)
    rTruth = np.linalg.norm(navOutputs.sim.rTruth_CA_A,axis=1)
    plt.axvline(x=17.68, color='red',alpha=0.5,linewidth=2)
    plt.axvline(x=np.min(rTruth)/1e3, color='blue', label='Data bounds',alpha=0.5,linewidth=2)
    plt.axvline(x=np.max(rTruth)/1e3, color='blue',alpha=0.5,linewidth=2)

    ax.set_xlim([rXYZ3DInterval[0,1]/1e3,50])
    ax.set_ylim([1e-2,1e3])
    plt.xlabel('Radius [km]', fontsize=font)
    plt.ylabel('Average gravity error [\%]', fontsize=font)
    ax.tick_params(axis='both', labelsize=font)
    ax.set_yscale('log')
    #ax.set_yticks([0.1,1,10,1e2,1e3,1e4,1e5,1e6,1e7])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid()
    ax.text(17.5, 4*1e2, '$r_{\mathrm{Brill}}$', va='center',fontsize=font,rotation=90,color='red')
    #ax.text(8.3, 2.5*1e-1, 'sphere', va='center',fontsize=font,rotation=90)
    ax.text(30, 4*1e2, '$r^{\mathrm{data}}_{\mathrm{min}}$', va='center',fontsize=font,rotation=90,color='blue')
    ax.text(38, 4*1e2, '$r^{\mathrm{data}}_{\mathrm{max}}$', va='center',fontsize=font,rotation=90,color='blue')
    lines = ax.get_lines()
    legend1 = pyplot.legend([lines[i] for i in [7]],
                            ["Keplerian"], loc=2, fontsize=font-4, bbox_to_anchor=(1.02, 1),borderaxespad=0.)
    legend2 = pyplot.legend([lines[i] for i in [0,1,2,3,4,5,6]],
                            ["SH deg=2", "SH deg=3","SH deg=4","SH deg=5","SH deg=6","SH deg=7","SH deg=8"], loc=2,
                            fontsize=font-4, bbox_to_anchor=(1.02, 0.887),borderaxespad=0.)
    legend3 = pyplot.legend([lines[i] for i in [8,11,14,17]],
                            ["M $n$=81","M $n$=161","M $n$=321","M $n$=641"], loc=2, fontsize=font-4,
                            bbox_to_anchor=(1.02, 0.3279),borderaxespad=0.)
    ax.add_artist(legend3)
    ax.add_artist(legend2)
    ax.add_artist(legend1)

    fig.savefig('Plots/AA/comparisonOnline.pdf', bbox_extra_artists=(legend1, legend2, legend3), bbox_inches='tight')


def plotGravErrSingle():
    # Do second plot
    #filenameSpherharm = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm5th.pck'
    filenameSpherharm = 'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm4th.pck'
    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/rand0.pck'
    navParamsSH, navOutputsSH = pickle.load(open(filenameSpherharm, "rb"))
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    rTruth = np.linalg.norm(navOutputsM.sim.rTruth_CA_A,axis=1)

    # Compute data bins
    nbinsTruth = 10
    rminTruth = np.min(rTruth)
    rmaxTruth = np.max(rTruth)
    binsTruth = np.zeros(nbinsTruth)

    # Filter truth values
    rTruthNavCam = rTruth*navOutputsM.sim.navCamSolution
    for ii in range(nbinsTruth):
        # Select data segment
        rminTruth_ii = rminTruth + ii*(rmaxTruth-rminTruth) / nbinsTruth
        rmaxTruth_ii = rminTruth + (ii+1)*(rmaxTruth-rminTruth) / nbinsTruth
        idx = np.where(np.logical_and(rTruthNavCam >= rminTruth_ii, rTruthNavCam <= rmaxTruth_ii))[0]
        binsTruth[ii] = len(idx)
    #binsTruth /= np.sum(binsTruth)

    # Compute error bins
    r = navOutputsM.sim.rXYZ3D
    aSHErr3D = navOutputsSH.nav.aErrXYZ3D
    aMErr3D = navOutputsM.nav.aErrXYZ3D
    a0Err3D = navOutputsM.nav.a0ErrXYZ3D
    r1D = np.ravel(r)
    aSHErr1D = np.ravel(aSHErr3D)
    aMErr1D = np.ravel(aMErr3D)
    a0Err1D = np.ravel(a0Err3D)
    r0 = np.nanmin(r1D)
    rf = np.nanmax(r1D)
    nbins = 80
    rLim = np.linspace(r0, rf + 1e-6 * rf, nbins + 1)
    rbins = rLim[0:-1] + (rLim[1] - rLim[0]) / 2
    aSHErrbins = np.zeros(nbins)
    aMErrbins = np.zeros(nbins)
    a0Errbins = np.zeros(nbins)
    contbins = np.zeros(nbins)
    for ii in range(len(r1D)):
        if np.isnan(r1D[ii]):
            continue
        dr = r1D[ii] - r0
        idx = int(np.floor(dr / (rLim[1] - rLim[0])))
        aSHErrbins[idx] += aSHErr1D[ii]
        aMErrbins[idx] += aMErr1D[ii]
        a0Errbins[idx] += a0Err1D[ii]
        contbins[idx] += 1
    aSHErrbins /= contbins
    aMErrbins /= contbins
    a0Errbins /= contbins

    plt.gcf()
    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(10, 1e10, marker='o', linestyle='', markersize=4, alpha=1, color=colors[2],label='SH deg=4')
    ax1.plot(10, 1e10, marker='o', linestyle='', markersize=4, alpha=1, color=colors[0],label='M $n$=641')
    ax1.plot(10, 1e10, marker='o', linestyle='', markersize=4, alpha=1, color='black',label='Keplerian')
    ax1.plot(r1D/1e3, aSHErr1D*100, marker='o', linestyle='', markersize=1, alpha=0.2, color=colors[2])
    ax1.plot(r1D/1e3, aMErr1D*100, marker='o', linestyle='', markersize=1, alpha=0.2, color=colors[0])
    ax1.plot(r1D/1e3, a0Err1D*100, marker='o', linestyle='', markersize=1, alpha=0.2, color='black')
    #ax1.plot(rbins/1e3, aSHErrbins*100, color=colors[0], linewidth=2)
    #ax1.plot(rbins/1e3, aMErrbins*100, color=colors[1], linewidth=2)
    #ax1.plot(rbins/1e3, a0Errbins*100, color=colors[2], linewidth=2)
    plt.axvline(x=np.min(rTruth)/1e3, color='blue',alpha=0.5,linewidth=2)
    plt.axvline(x=np.max(rTruth)/1e3, color='blue',alpha=0.5,linewidth=2)
    plt.axvline(x=17.68, color='red',alpha=0.5,linewidth=2)
    ax1.text(17.5, 3*1e2, '$r_{\mathrm{Brill}}$', va='center',fontsize=font,rotation=90,color='red')
    #ax.text(8.3, 2.5*1e-1, 'sphere', va='center',fontsize=font,rotation=90)
    ax1.text(30, 3*1e2, '$r^{\mathrm{data}}_{\mathrm{min}}$', va='center',fontsize=font,rotation=90,color='blue')
    ax1.text(38, 3*1e2, '$r^{\mathrm{data}}_{\mathrm{max}}$', va='center',fontsize=font,rotation=90,color='blue')

    ax1.set_yscale('log')
    ax1.set_ylim([1e-3,1e3])
    ax1.set_yticks([1e-2,1e-1,0.1,1,1e1,1e2,1e3])

    #ax1.text(7.5, 5*1e-2, 'Brillouin', va='center',fontsize=font)
    #ax1.text(8.3, 1.8*1e-2, 'sphere', va='center',fontsize=font)
    #ax1.text(31, 4.5*1e4, 'In-orbit', va='center',fontsize=font)
    #ax1.text(32, 1.5*1e4, 'data', va='center',fontsize=font)

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=font-4,borderaxespad=0.22,loc='lower left')

    plt.xlabel('Radius [km]', fontsize=font)
    plt.ylabel('Gravity error [\%]', fontsize=font)
    ax1.tick_params(axis='both', labelsize=font)

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax1.yaxis.set_minor_locator(y_minor)
    ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax1.grid()
    ax1.set_xlim([np.nanmin(r1D/1e3),50])

    secax_y = ax1.twinx()
    for ii in range(nbinsTruth):
        secax_y.bar(rminTruth/1e3+(ii+0.5)*(rmaxTruth-rminTruth)/nbinsTruth/1e3, binsTruth[ii],
                (rmaxTruth-rminTruth)/nbinsTruth/1e3, 0, alpha=0.1, color='black')
    #secax_y.set_yticks([])
    secax_y.set_ylim([0,1800])
    secax_y.set_yticks([0,300,600,900,1200,1500,1800])
    secax_y.tick_params(axis='y', labelsize=font)
    secax_y.set_ylabel('Data samples [-]',fontsize=font)

    fig1.savefig('Plots/AA/comparisonOnlineIndividual.pdf', bbox_inches='tight')

def plotGravErr2D():
    filenameSpherharm = 'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm4th.pck'
    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/rand0.pck'
    navParamsSH, navOutputsSH = pickle.load(open(filenameSpherharm, "rb"))
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))

    # Plot the global gravity results
    vertList, faceList, nVertex, nFacet = loadPolyFromFileToList(polyFilename)
    xyzPoly = np.array(vertList)
    color_smallbody = [105 / 255, 105 / 255, 105 / 255]

    rTruth = navOutputsM.sim.rTruth_CA_A
    Xy2D = navOutputsM.sim.Xy2D
    Yx2D = navOutputsM.sim.Yx2D
    Xz2D = navOutputsM.sim.Xz2D
    Zx2D = navOutputsM.sim.Zx2D
    Yz2D = navOutputsM.sim.Yz2D
    Zy2D = navOutputsM.sim.Zy2D
    accMErrXY2D = navOutputsM.nav.aErrXY2D
    accMErrXZ2D = navOutputsM.nav.aErrXZ2D
    accMErrYZ2D = navOutputsM.nav.aErrYZ2D
    accSHErrXY2D = navOutputsSH.nav.aErrXY2D
    accSHErrXZ2D = navOutputsSH.nav.aErrXZ2D
    accSHErrYZ2D = navOutputsSH.nav.aErrYZ2D
    acc0ErrXY2D = navOutputsM.nav.a0ErrXY2D
    acc0ErrXZ2D = navOutputsM.nav.a0ErrXZ2D
    acc0ErrYZ2D = navOutputsM.nav.a0ErrYZ2D
    acc0ErrYZ2D[0,0] = 0
    accMErrXY2D[accMErrXY2D > 0.1] = 0.1
    accMErrXZ2D[accMErrXZ2D > 0.1] = 0.1
    accMErrYZ2D[accMErrYZ2D > 0.1] = 0.1
    acc0ErrXY2D[acc0ErrXY2D > 0.1] = 0.1
    acc0ErrXZ2D[acc0ErrXZ2D > 0.1] = 0.1
    acc0ErrYZ2D[acc0ErrYZ2D > 0.1] = 0.1
    accSHErrXY2D[accSHErrXY2D > 0.1] = 0.1
    accSHErrXZ2D[accSHErrXZ2D > 0.1] = 0.1
    accSHErrYZ2D[accSHErrYZ2D > 0.1] = 0.1
    font = 20
    plt.gcf()
    fig3, ax3 = plt.subplots(3,3, figsize=(12,9), gridspec_kw={'width_ratios': [1, 1, 1]})

    ticks = np.linspace(0, 0.10, 11)*100
    minX = -50*1e3
    maxX = 50*1e3

    cs = ax3[0,0].contourf(Xy2D/1e3, Yx2D/1e3, acc0ErrXY2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[0,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
    ax3[0,0].set_ylabel('$y$ [km]', fontsize=font)
    ax3[0,0].tick_params(axis='both', labelsize=font)
    ax3[0,0].set_xlim([minX/1e3, maxX/1e3])
    ax3[0,0].set_ylim([minX/1e3, maxX/1e3])
    ax3[0,0].set_xticks([-50,-25,0,25,50])
    ax3[0,0].set_yticks([-50,-25,0,25,50])
    #ax3[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    #ax3[0,0].yaxis.set_minor_locator(AutoMinorLocator())
    ax3[0,0].set_xticklabels([])

    cs = ax3[0,1].contourf(Xz2D/1e3, Zx2D/1e3, acc0ErrXZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[0,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[0,1].set_ylabel('$z$ [km]', fontsize=font)
    ax3[0,1].tick_params(axis='both', labelsize=font)
    ax3[0,1].set_title('Keplerian', fontsize=font, fontweight='bold')
    ax3[0,1].set_xlim([minX/1e3, maxX/1e3])
    ax3[0,1].set_ylim([minX/1e3, maxX/1e3])
    ax3[0,1].set_xticks([-50,-25,0,25,50])
    ax3[0,1].set_xticklabels([])
    ax3[0,1].set_yticklabels([])

    cs = ax3[0,2].contourf(Yz2D/1e3, Zy2D/1e3, acc0ErrYZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[0,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[0,2].plot()
    #cbar = fig3.colorbar(cs, ax=ax3[0,2], shrink=0.95, ticks=ticks)
    #cbar.ax.set_yticklabels(['0', '', '2', '', '4', '', '6', '', '8', '', '$>$10', ''])
    #cbar.ax.tick_params(labelsize=font)
    #cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font)
    ax3[0,2].set_ylabel('$z$ [km]', fontsize=font)
    ax3[0,2].tick_params(axis='both', labelsize=font)
    ax3[0,2].set_xlim([minX/1e3, maxX/1e3])
    ax3[0,2].set_ylim([minX/1e3, maxX/1e3])
    ax3[0,2].set_xticks([-50,-25,0,25,50])
    ax3[0,2].set_xticklabels([])
    ax3[0,2].set_yticklabels([])

    cs = ax3[1,0].contourf(Xy2D/1e3, Yx2D/1e3, accSHErrXY2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[1,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
    ax3[1,0].plot(rTruth[:,0]/1e3, rTruth[:,1]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax3[1,0].set_ylabel('$y$ [km]', fontsize=font)
    ax3[1,0].tick_params(axis='both', labelsize=font)
    ax3[1,0].set_xlim([minX/1e3, maxX/1e3])
    ax3[1,0].set_ylim([minX/1e3, maxX/1e3])
    ax3[1,0].set_xticks([-50,-25,0,25,50])
    ax3[1,0].set_yticks([-50,-25,0,25,50])
    ax3[1,0].set_xticklabels([])

    cs = ax3[1,1].contourf(Xz2D/1e3, Zx2D/1e3, accSHErrXZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[1,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[1,1].plot(rTruth[:,0]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax3[1,1].set_ylabel('$z$ [km]', fontsize=font)
    ax3[1,1].tick_params(axis='both', labelsize=font)
    ax3[1,1].set_title('Spherical harmonics deg=4', fontsize=font)
    ax3[1,1].set_xlim([minX/1e3, maxX/1e3])
    ax3[1,1].set_ylim([minX/1e3, maxX/1e3])
    ax3[1,1].set_xticks([-50,-25,0,25,50])
    ax3[1,1].set_xticklabels([])
    ax3[1,1].set_yticklabels([])

    cs = ax3[1,2].contourf(Yz2D/1e3, Zy2D/1e3, accSHErrYZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[1,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[1,2].plot(rTruth[:,1]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    #cbar = fig3.colorbar(cs, ax=ax3[1,2], shrink=0.95, ticks=ticks, extend='max')
    #cbar.ax.set_yticklabels(['0', '', '2', '', '4', '', '6', '', '8', '', '$>$10', ''])
    #cbar.ax.tick_params(labelsize=font)
    #cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font)
    ax3[1,2].set_ylabel('$z$ [km]', fontsize=font)
    ax3[1,2].tick_params(axis='both', labelsize=font)
    ax3[1,2].set_xlim([minX/1e3, maxX/1e3])
    ax3[1,2].set_ylim([minX/1e3, maxX/1e3])
    ax3[1,2].set_xticks([-50,-25,0,25,50])
    ax3[1,2].set_xticklabels([])
    ax3[1,2].set_yticklabels([])

    cs = ax3[2,0].contourf(Xy2D/1e3, Yx2D/1e3, accMErrXY2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[2,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
    ax3[2,0].plot(rTruth[:,0]/1e3, rTruth[:,1]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax3[2,0].set_xlabel('$x$ [km]', fontsize=font)
    ax3[2,0].set_ylabel('$y$ [km]', fontsize=font)
    ax3[2,0].tick_params(axis='both', labelsize=font)
    ax3[2,0].set_xlim([minX/1e3, maxX/1e3])
    ax3[2,0].set_ylim([minX/1e3, maxX/1e3])
    ax3[2,0].set_xticks([-50,-25,0,25,50])
    ax3[2,0].set_yticks([-50,-25,0,25,50])

    cs = ax3[2,1].contourf(Xz2D/1e3, Zx2D/1e3, accMErrXZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"))
    ax3[2,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[2,1].plot(rTruth[:,0]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    ax3[2,1].set_xlabel('$x$ [km]', fontsize=font)
    ax3[2,1].set_ylabel('$z$ [km]', fontsize=font)
    ax3[2,1].tick_params(axis='both', labelsize=font)
    ax3[2,1].set_title('Mascon $n$=641', fontsize=font)
    ax3[2,1].set_xlim([minX/1e3, maxX/1e3])
    ax3[2,1].set_ylim([minX/1e3, maxX/1e3])
    ax3[2,1].set_xticks([-50,-25,0,25,50])
    ax3[2,1].set_yticklabels([])

    cs = ax3[2,2].contourf(Yz2D/1e3, Zy2D/1e3, accMErrYZ2D*100, levels=100, vmax=10, cmap=get_cmap("viridis"), extend='max')
    ax3[2,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
    ax3[2,2].plot(rTruth[:,1]/1e3, rTruth[:,2]/1e3, color='white',linestyle='dotted',linewidth=0.5)
    #cbar = fig3.colorbar(cs, ax=ax3[2,2], shrink=0.95, ticks=ticks)
    #cbar.ax.set_yticklabels(['0', '', '2', '', '4', '', '6', '', '8', '', '$>$10', ''])
    #cbar.ax.tick_params(labelsize=font)
    #cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font)
    ax3[2,2].set_xlabel('$y$ [km]', fontsize=font)
    ax3[2,2].set_ylabel('$z$ [km]', fontsize=font)
    ax3[2,2].tick_params(axis='both', labelsize=font)
    ax3[2,2].set_xlim([minX/1e3, maxX/1e3])
    ax3[2,2].set_ylim([minX/1e3, maxX/1e3])
    ax3[2,2].set_xticks([-50,-25,0,25,50])
    ax3[2,2].set_yticklabels([])

    fig3.subplots_adjust(right=0.8)
    cbar_ax = fig3.add_axes([0.825, 0.2, 0.02, 0.6])
    cbar = fig3.colorbar(cs, cax=cbar_ax, shrink=0.95, extend='max', ticks=ticks)
    cbar.set_label('Gravity error [\%]', fontsize=font, rotation=90)
    cbar.ax.set_yticklabels(['$0$', '', '$2$', '', '$4$', '', '$6$', '', '$8$', '', '$\geq$$10$'])
    cbar.ax.tick_params(labelsize=font)

    #ax3[1,0].set_rasterized(True)
    #ax3[1,1].set_rasterized(True)
    #ax3[1,2].set_rasterized(True)
    #ax3[2,0].set_rasterized(True)
    #ax3[2,1].set_rasterized(True)
    #ax3[2,2].set_rasterized(True)

    plt.savefig('Plots/AA/gravity2Dcomparison.pdf', format='pdf')

def plotMascon():
    ################################### Mascons plot ###################################################################
    vertList, faceList, nVertex, nFacet = loadPolyFromFileToList(polyFilename)
    xyzPoly = np.array(vertList)
    idxPoly = np.array(faceList)

    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons640sqrLAM1E-1WinitV2/rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    posM = navOutputsM.nav.posMEst
    muM = navOutputsM.nav.muMEst

    font = 18
    # Get polyhedron and landmarks
    color_smallbody = [105/255, 105/255, 105/255]

    # Plot the relative 3D position
    plt.gcf()
    fig2 = plt.figure(figsize=(7,7))
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')

    ax2.plot_trisurf(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, triangles=idxPoly-1, color=color_smallbody, zorder=0, alpha=.2)
    p = ax2.scatter3D(posM[:,0]/1e3, posM[:,1]/1e3, posM[:,2]/1e3, c=muM, cmap=get_cmap("viridis"))
    cbar = fig2.colorbar(p, ax=ax2, shrink=0.7,extend='min')#, ticks=[-15000, -10000, -5000, 0, 5000, 10000, 15000, 20000])
    cbar.formatter.set_powerlimits((0,0))
    cbar.formatter.set_useMathText(True)
    cbar.ax.tick_params(labelsize=font)
    cbar.ax.yaxis.get_offset_text().set_fontsize(font)
    cbar.ax.yaxis.set_offset_position('left')
    #cbar.ax.set_yticklabels(['-2.0·10$^4$', '-1.5·10$^4$', '-1.0·10$^4$', '-0.5·10$^4$', '0', '0.5·10$^4$',
    #                         '1.0·10$^4$', '1.5·10$^4$', '2.0·10$^4$'])
    cbar.set_label('$\mu_M$ [m$^3$/s$^2$]', rotation=90, fontsize=font)
    ax2.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax2.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    ax2.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax2.tick_params(axis='both', labelsize=font)
    ax2.set_facecolor('white')

    #x_minor = mpl.ticker.LogLocator(numticks=5)
    #y_minor = mpl.ticker.LogLocator(numticks=5)
    #z_minor = mpl.ticker.LogLocator(numticks=5)
    #ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    #ax.zaxis.set_minor_locator(AutoMinorLocator(z_minor))

    set_axes_equal(ax2)
    ax2.view_init(elev=59., azim=360-144)
    plt.savefig('Plots/AA/mascons640.pdf', format='pdf')
    # plt.savefig('Plots/TAES/mascons.pdf', format='pdf', bbox_inches='tight')

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
    filenameMascons = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons160sqrLAM1E-1Winit/rand0.pck'
    navParamsM, navOutputsM = pickle.load(open(filenameMascons, "rb"))
    posM = navOutputsM.nav.posMEst
    Nsector = int(len(posM)/8)


    cmap = mpl.cm.get_cmap("viridis")
    bottom = cmap(0)
    top = cmap(cmap.N)
    fig6 = plt.figure(figsize=(6,6))
    ax6 = fig6.add_subplot(1, 1, 1, projection='3d')
    #ax.set_aspect('equal')
    ax6.plot_surface(Xxy, Yxy, Zxy, zorder=1, alpha=0.05, color='blue')
    ax6.plot_surface(Xxz, Yxz, Zxz, zorder=1, alpha=0.05, color='blue')
    ax6.plot_surface(Xyz, Yyz, Zyz, zorder=1, alpha=0.05, color='blue')
    ax6.plot_trisurf(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, triangles=idxPoly-1, color=color_smallbody, zorder=10, alpha=.4)
    for ii in range(8):
        if ii == 0:
            ax6.plot(posM[ii*Nsector+1:(ii+1)*Nsector+1,0]/1e3, posM[ii*Nsector+1:(ii+1)*Nsector+1,1]/1e3,
                     posM[ii*Nsector+1:(ii+1)*Nsector+1,2]/1e3, color=bottom, linestyle='', marker='.', markersize=7.5, zorder=20,label='$\mu_{M_k}=0$')
        else:
            ax6.plot(posM[ii*Nsector+1:(ii+1)*Nsector+1,0]/1e3, posM[ii*Nsector+1:(ii+1)*Nsector+1,1]/1e3,
                     posM[ii*Nsector+1:(ii+1)*Nsector+1,2]/1e3, color=bottom, linestyle='', marker='.', markersize=7.5, zorder=20)
    ax6.plot(0,0,0,color=top,linestyle='',marker='.',markersize=10, zorder=20,label='$\mu_{M_n}=\mu$')
    #ax.set_xlabel('$x$ [km]', fontsize=14)
    #ax.set_ylabel('$y$ [km]', fontsize=14)
    #ax.set_zlabel('$z$ [km]', fontsize=14)
    ax6.grid(False)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_zticks([])
    plt.axis('off')

    #ax.set_title('Small body centred fixed frame', fontsize=16)
    ax6.tick_params(axis='both', labelsize=14)
    ax6.set_facecolor('white')
    ax6.legend(fontsize=18,bbox_to_anchor=(0.76,0.835))
    #ax6.legend(fontsize=font,bbox_to_anchor=(1.2,0.925))

    set_axes_equal(ax6)
    ax6.view_init(elev=59., azim=360-144)
    #legend1 = pyplot.legend([lines[i] for i in [0,1000]],
    #                        ["$\mu_{M_k}=0$","$\mu_{M_n}=\mu$"], loc=2, fontsize=font-4, bbox_to_anchor=(1.2, 0.925),borderaxespad=0.)
    #ax6.add_artist(legend1)
    plt.tight_layout()
    plt.savefig('Plots/AA/mascons160Init.pdf', format='pdf', bbox_inches='tight')


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

if __name__ == "__main__":
    polyFilename = '/Users/julio/basilisk/supportData/LocalGravData/eros007790.tab'
    font = 22
    colors = list(mcolors.TABLEAU_COLORS.values())
    #plotGravErrMean()
    #plotGravErrSingle()
    #plotGravErr2D()
    plotMascon()
    #plotMascon0()
    #plotNavCamErr()
    plotPropRMS()
    #plotDMCUKF()
    plt.show()


