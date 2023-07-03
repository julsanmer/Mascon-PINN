import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse

# --------------------------------- COMPONENTS & SUBPLOT HANDLING ----------------------------------------------- #
color_x = 'dodgerblue'
color_y = 'salmon'
color_z = 'lightgreen'
m2km = 1.0 / 1000.0

def show_all_plots():
    plt.show()

def clear_all_plots():
    plt.close("all")

def save_all_plots(fileName, figureNames):
    figureList = {}
    numFigures = len(figureNames)
    for i in range(0, numFigures):
        pltName = fileName + "_" + figureNames[i]
        figureList[pltName] = plt.figure(i)
    return figureList


# ------------------------------------- MAIN PLOT HANDLING ------------------------------------------------------ #

def plot_RMS(scenario1Out, scenario2Out):
    r1RMS = scenario1Out.rRMS
    v1RMS = scenario1Out.vRMS
    a1RMS = scenario1Out.aRMS
    r2RMS = scenario2Out.rRMS
    v2RMS = scenario2Out.vRMS
    a2RMS = scenario2Out.aRMS

    """Plot the RMS results."""
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(r1RMS, 'b', marker='o', linestyle='--', label='UKF')
    ax[1].plot(v1RMS, 'b', marker='o', linestyle='--')
    ax[2].plot(a1RMS*1e3, 'b', marker='o',  linestyle='--')

    ax[0].plot(r2RMS, color='orange', marker='o', linestyle='--', label='HOU')
    ax[1].plot(v2RMS, color='orange', linestyle='--', marker='o')
    ax[2].plot(a2RMS*1e3, color='orange', linestyle='--', marker='o')

    plt.xlabel('Training steps [-]')

    ax[0].set_ylabel('Position RMS [m]')
    ax[1].set_ylabel('Velocity RMS [m/s]')
    ax[2].set_ylabel('Acceleration RMS [mm/s$^2$]')

    ax[0].legend()

def plot_rnormErr(time, rTruth, r1Est, r2Est, r3Est):
    """Plot the position error of propagation."""
    nPoints = len(time)
    r1Err = r1Est[1:nPoints+1,0:3] - rTruth
    r1normErr = np.linalg.norm(r1Err, axis=1)
    r2Err = r2Est[1:nPoints+1,0:3] - rTruth
    r2normErr = np.linalg.norm(r2Err, axis=1)
    r3Err = r3Est[1:nPoints+1,0:3] - rTruth
    r3normErr = np.linalg.norm(r3Err, axis=1)

    plt.gcf()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(time[1:nPoints+1]/(24*3600), r1normErr/1e3, 'b', label='UKF')
    ax.plot(time[1:nPoints+1]/(24*3600), r2normErr/1e3, color='orange', label='HOU')
    ax.plot(time[1:nPoints+1]/(24*3600), r3normErr/1e3, color='green', label='Kep.')

    plt.xlabel('Time [days]')
    plt.ylabel('Position error [km]')

    ax.legend(loc='upper left')

def plot_rRMSErr(time, rTruth, r1Est, r2Est, r3Est):
    """Plot the position error of propagation."""
    nPoints = len(time)
    r1Err = r1Est[1:nPoints+1,0:3] - rTruth
    r1normErr = np.linalg.norm(r1Err, axis=1)
    r2Err = r2Est[1:nPoints+1,0:3] - rTruth
    r2normErr = np.linalg.norm(r2Err, axis=1)
    r3Err = r3Est[1:nPoints+1,0:3] - rTruth
    r3normErr = np.linalg.norm(r3Err, axis=1)

    r1RMS = np.zeros(nPoints-1)
    r2RMS = np.zeros(nPoints-1)
    r3RMS = np.zeros(nPoints-1)
    sumRMS1 = 0
    sumRMS2 = 0
    sumRMS3 = 0
    for ii in range(nPoints-1):
        sumRMS1 += (r1normErr[ii])**2
        sumRMS2 += (r2normErr[ii])**2
        sumRMS3 += (r3normErr[ii])**2

        r1RMS[ii] = np.sqrt(sumRMS1/(ii+1))
        r2RMS[ii] = np.sqrt(sumRMS2/(ii+1))
        r3RMS[ii] = np.sqrt(sumRMS3/(ii+1))

    plt.gcf()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(time[1:nPoints+1]/(24*3600), r1RMS/1e3, 'b', label='UKF')
    ax.plot(time[1:nPoints+1]/(24*3600), r2RMS/1e3, color='orange', label='HOU')
    ax.plot(time[1:nPoints+1]/(24*3600), r3RMS/1e3, color='green', label='Kep.')

    plt.xlabel('Time [days]')
    plt.ylabel('Position RMS [km]')

    ax.legend(loc='upper left')


def plot_globalGravity2D(scenario1Out, scenario2Out, scenarioParams, title1, title2, plotPoly=True, figName=[],
                         smallbody=[], comparison=False):
    # Unfold data
    X1 = scenario1Out.sim.X2D
    Y1 = scenario1Out.sim.Y2D
    Zx1 = scenario1Out.sim.Zx2D
    Zy1 = scenario1Out.sim.Zy2D
    aXYErr1 = scenario1Out.nav.aErrXY2D
    aXZErr1 = scenario1Out.nav.aErrXZ2D
    aYZErr1 = scenario1Out.nav.aErrYZ2D

    if comparison:
        aXYErr1[aXYErr1 > 0.25] = 0.25
        aXZErr1[aXZErr1 > 0.25] = 0.25
        aYZErr1[aYZErr1 > 0.25] = 0.25
        aXYErr1[aXYErr1 < -0.25] = -0.25
        aXZErr1[aXZErr1 < -0.25] = -0.25
        aYZErr1[aYZErr1 < -0.25] = -0.25
    else:
        aXYErr1[aXYErr1 > 0.5] = 0.55
        aXZErr1[aXZErr1 > 0.5] = 0.55
        aYZErr1[aYZErr1 > 0.5] = 0.55

    rTruth1 = scenario1Out.sim.rTruth_CA_A

    minX1 = -scenario1Out.sim.hmaxPlot
    maxX1 = scenario1Out.sim.hmaxPlot

    linestyle = 'solid'#'dashed'
    color_smallbody = [105 / 255, 105 / 255, 105 / 255]

    # Generate ellipses
    if plotPoly:
        xyzPoly = scenarioParams.smallbody.xyzPoly
    else:
        axes = scenarioParams.smallbody.axesEllipsoid
        aEll = axes[0]
        bEll = axes[1]
        cEll = axes[2]
        ellXY1 = Ellipse(xy=(0,0), width=2*aEll/1e3, height=2*bEll/1e3, angle=0)
        ellXZ1 = Ellipse(xy=(0,0), width=2*aEll/1e3, height=2*cEll/1e3, angle=0)
        ellYZ1 = Ellipse(xy=(0,0), width=2*bEll/1e3, height=2*cEll/1e3, angle=0)
        ellXY2 = Ellipse(xy=(0,0), width=2*aEll/1e3, height=2*bEll/1e3, angle=0)
        ellXZ2 = Ellipse(xy=(0,0), width=2*aEll/1e3, height=2*cEll/1e3, angle=0)
        ellYZ2 = Ellipse(xy=(0,0), width=2*bEll/1e3, height=2*cEll/1e3, angle=0)

    if scenario2Out:
        X2 = scenario2Out.X2D
        Y2 = scenario2Out.Y2D
        Zx2 = scenario2Out.Zx2D
        Zy2 = scenario2Out.Zy2D


        aXYErr2 = scenario2Out.aXYErr2D
        aXZErr2 = scenario2Out.aXZErr2D
        aYZErr2 = scenario2Out.aYZErr2D

        aXYErr2[aXYErr2 > 1] = 1
        aXZErr2[aXZErr2 > 1] = 1
        aYZErr2[aYZErr2 > 1] = 1

        rTruth2 = scenario2Out.rTruth_CA_A

        minX2 = -scenario2Out.hmaxPlot
        maxX2 = scenario2Out.hmaxPlot

        """Plot the global gravity results."""
        plt.gcf()
        fig, ax = plt.subplots(2, 3, sharex=True, figsize=(12,6), gridspec_kw={'width_ratios': [1,1,1.2]})
        #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ticks = np.linspace(0,1,11)

        cs = ax[0,0].contourf(X1/1e3, Y1/1e3, aXYErr1, levels=ticks, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[0,0], shrink=0.9, ticks=ticks)
        #cbar.ax.set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','>0.9',''])
        if plotPoly:
            ax[0,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
        else:
            ax[0,0].add_artist(ellXY1)
            ellXY1.set_clip_box(ax[0, 0].bbox)
            ellXY1.set_alpha(1)
            ellXY1.set_facecolor(color_smallbody)
        ax[0,0].plot(rTruth1[:,0]/1e3, rTruth1[:,1]/1e3,'k', linestyle=linestyle)
        ax[0,0].set_xlim([minX1/1e3,maxX1/1e3])
        ax[0,0].set_ylim([minX1/1e3,maxX1/1e3])
        #ax[0,0].set_xlabel('$x^A$ [km]')
        ax[0,0].set_ylabel('$y^A$ [km]')
        ax[0,0].set_title(title1)
        #ax[0].axis('scaled')

        cs = ax[0,1].contourf(X1/1e3, Zx1/1e3, aXZErr1, levels=ticks, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[0,1], shrink=0.9, ticks=ticks)
        #cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[0,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[0,1].add_artist(ellXZ1)
            ellXZ1.set_clip_box(ax[0,1].bbox)
            ellXZ1.set_alpha(1)
            ellXZ1.set_facecolor(color_smallbody)
        ax[0,1].plot(rTruth1[:,0]/1e3, rTruth1[:,2]/1e3, 'k', linestyle=linestyle)
        ax[0,1].set_xlim([minX1/1e3,maxX1/1e3])
        ax[0,1].set_ylim([minX1/1e3,maxX1/1e3])
        #ax[0,1].set_xlabel('$x^A$ [km]')
        ax[0,1].set_ylabel('$z^A$ [km]')
        ax[0,1].set_title(title1)
        #ax[1].axis('scaled')

        cs = ax[0,2].contourf(Y1/1e3, Zy1/1e3, aYZErr1, levels=ticks, cmap=get_cmap("jet"))
        cbar = fig.colorbar(cs, ax=ax[0,2], shrink=0.9, ticks=ticks)
        cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[0,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[0,2].add_artist(ellYZ1)
            ellYZ1.set_clip_box(ax[0,2].bbox)
            ellYZ1.set_alpha(1)
            ellYZ1.set_facecolor(color_smallbody)
        ax[0,2].plot(rTruth1[:,1]/1e3, rTruth1[:,2]/1e3,'k', linestyle=linestyle)
        ax[0,2].set_xlim([minX1/1e3,maxX1/1e3])
        ax[0,2].set_ylim([minX1/1e3,maxX1/1e3])
        #ax[0,2].set_xlabel('$y^A$ [km]')
        ax[0,2].set_ylabel('$z^A$ [km]')
        ax[0,2].set_title(title1)
        #ax[2].axis('scaled')

        cs = ax[1,0].contourf(X2/1e3, Y2/1e3, aXYErr2, levels=ticks, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[1,0], shrink=0.9, ticks=ticks)
        #cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[1,0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
        else:
            ax[1,0].add_artist(ellXY2)
            ellXY2.set_clip_box(ax[1,0].bbox)
            ellXY2.set_alpha(1)
            ellXY2.set_facecolor(color_smallbody)
        ax[1,0].plot(rTruth2[:,0]/1e3, rTruth2[:,1]/1e3, 'k', linestyle=linestyle)
        ax[1,0].set_xlim([minX2/1e3,maxX2/1e3])
        ax[1,0].set_ylim([minX2/1e3,maxX2/1e3])
        ax[1,0].set_xlabel('$x^A$ [km]')
        ax[1,0].set_ylabel('$y^A$ [km]')
        ax[1,0].set_title(title2)
        #ax[0].axis('scaled')

        cs = ax[1,1].contourf(X2/1e3, Zx2/1e3, aXZErr2, levels=ticks, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[1,1], shrink=0.9, ticks=ticks)
        #cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[1,1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[1,1].add_artist(ellXZ2)
            ellXZ2.set_clip_box(ax[1,1].bbox)
            ellXZ2.set_alpha(1)
            ellXZ2.set_facecolor(color_smallbody)
        ax[1,1].plot(rTruth2[:,0]/1e3, rTruth2[:,2]/1e3, 'k', linestyle=linestyle)
        ax[1,1].set_xlim([minX2/1e3,maxX2/1e3])
        ax[1,1].set_ylim([minX2/1e3,maxX2/1e3])
        ax[1,1].set_xlabel('$x^A$ [km]')
        ax[1,1].set_ylabel('$z^A$ [km]')
        ax[1,1].set_title(title2)
        #ax[1].axis('scaled')

        cs = ax[1,2].contourf(Y2/1e3, Zy2/1e3, aYZErr2, levels=ticks, cmap=get_cmap("jet"))
        cbar = fig.colorbar(cs, ax=ax[1,2], shrink=0.9, ticks=ticks)
        cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[1,2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[1,2].add_artist(ellYZ2)
            ellYZ2.set_clip_box(ax[1,2].bbox)
            ellYZ2.set_alpha(1)
            ellYZ2.set_facecolor(color_smallbody)
        ax[1,2].plot(rTruth2[:,1]/1e3, rTruth2[:,2]/1e3, 'k', linestyle=linestyle)
        ax[1,2].set_xlim([minX2/1e3,maxX2/1e3])
        ax[1,2].set_ylim([minX2/1e3,maxX2/1e3])
        ax[1,2].set_xlabel('$y^A$ [km]')
        ax[1,2].set_ylabel('$z^A$ [km]')
        ax[1,2].set_title(title2)
        #ax[2].axis('scaled')

        plt.tight_layout()

        if figName:
            figName = 'Plots/' + smallbody + '_' + figName + '2Dgravity' + '.eps'
            plt.savefig(figName, format='eps')
    else:
        """Plot the global gravity results."""
        plt.gcf()
        #fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7.5,3), gridspec_kw={'width_ratios': [1, 1.2]})
        fig, ax = plt.subplots(1, 3, sharex=True, figsize=(12,3.65), gridspec_kw={'width_ratios': [1,1,1.23]})
        #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        if comparison:
            ticks = np.linspace(-0.25, 0.25, 11)
        else:
            ticks = np.linspace(0,0.55,12)

        cs = ax[0].contourf(X1/1e3, Y1/1e3, aXYErr1, levels=100, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[0], shrink=0.9, ticks=ticks)
        if plotPoly:
            ax[0].plot(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, color=color_smallbody)
        else:
            ax[0].add_artist(ellXY1)
            ellXY1.set_clip_box(ax[0].bbox)
            ellXY1.set_alpha(1)
            ellXY1.set_facecolor(color_smallbody)
        ax[0].plot(rTruth1[:,0]/1e3, rTruth1[:,1]/1e3, 'k', linestyle=linestyle, linewidth=0.5)
        ax[0].set_xlim([minX1/1e3,maxX1/1e3])
        ax[0].set_ylim([minX1/1e3,maxX1/1e3])
        ax[0].set_xlabel('$x^A$ [km]',fontsize=12)
        ax[0].set_ylabel('$y^A$ [km]',fontsize=12)
        ax[0].tick_params(axis='both', labelsize=12)
        #ax[0].axis('scaled')

        cs = ax[1].contourf(X1/1e3, Zx1/1e3, aYZErr1, levels=100, cmap=get_cmap("jet"))
        #cbar = fig.colorbar(cs, ax=ax[1], shrink=0.9, ticks=ticks)
        #cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '>0.9', ''])
        if plotPoly:
            ax[1].plot(xyzPoly[:,0]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[1].add_artist(ellXZ1)
            ellYZ1.set_clip_box(ax[1].bbox)
            ellYZ1.set_alpha(1)
            ellYZ1.set_facecolor(color_smallbody)
        ax[1].plot(rTruth1[:,0]/1e3, rTruth1[:,2]/1e3, 'k', linestyle=linestyle, linewidth=0.5)
        ax[1].set_xlim([minX1/1e3,maxX1/1e3])
        ax[1].set_ylim([minX1/1e3,maxX1/1e3])
        ax[1].set_xlabel('$x^A$ [km]', fontsize=12)
        ax[1].set_ylabel('$z^A$ [km]', fontsize=12)
        ax[1].tick_params(axis='both', labelsize=12)
        #ax[1].axis('scaled')

        cs = ax[2].contourf(Y1/1e3, Zy1/1e3, aYZErr1, levels=100, cmap=get_cmap("jet"))
        cbar = fig.colorbar(cs, ax=ax[2], shrink=0.95, ticks=ticks)
        if comparison:
            cbar.ax.set_yticklabels(['-0.25','-0.20','-0.15','-0.10','-0.05','0','0.05','0.10','0.15','0.20','0.25'])
        else:
            cbar.ax.set_yticklabels(['0','0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','>0.50',''])
        cbar.ax.tick_params(labelsize=12)
        if plotPoly:
            ax[2].plot(xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, color=color_smallbody)
        else:
            ax[2].add_artist(ellYZ1)
            ellYZ1.set_clip_box(ax[2].bbox)
            ellYZ1.set_alpha(1)
            ellYZ1.set_facecolor(color_smallbody)
        ax[2].plot(rTruth1[:,1]/1e3, rTruth1[:,2]/1e3,'k', linestyle=linestyle, linewidth=0.5)
        ax[2].set_xlim([minX1/1e3,maxX1/1e3])
        ax[2].set_ylim([minX1/1e3,maxX1/1e3])
        ax[2].set_xlabel('$y^A$ [km]', fontsize=12)
        ax[2].set_ylabel('$z^A$ [km]', fontsize=12)
        ax[2].tick_params(axis='both', labelsize=12)
        #ax[2].axis('scaled')

        #plt.savefig('Eros_error.eps', format='eps')


def plot_globalGravity3D(scenario1Out, scenario2Out, legend1, legend2, figName=[], smallbody=[]):
    a1Err3D = np.ravel(scenario1Out.aErr3D)
    a2Err3D = np.ravel(scenario2Out.aErr3D)

    #sha = scenario1Out.aErr3D.shape
    #for ii in range(sha[0]):
    #    for jj in range(sha[1]):
    #        for kk in range(sha[2]):
    #            if scenario1Out.aErr3D[ii,jj,kk] > 0.99 and scenario1Out.aErr3D[ii,jj,kk] < 1.01:
    #                print([scenario1Out.aErr3D[ii,jj,kk],[ii,jj,kk]])


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(a1Err3D, bins=np.linspace(0, 2, 40), stat='probability', kde=False, element="step", color='blue',alpha=0.1)
    sns.histplot(a2Err3D, bins=np.linspace(0, 2, 40), stat='probability', kde=False, element="step", color='orange',alpha=0.1)
    ax.set_xlabel('Acceleration error [-]')
    ax.set_ylabel('p [-]')
    ax.legend([legend1,legend2])
    plt.tight_layout()
    if figName:
        figName = 'Plots/' + smallbody + '_' + figName + '3Dgravity' + '.png'
        plt.savefig(figName, format='png')


def plot_globalPosition(rTruth, r1, r2, r3, axes):
    """Plot the global gravity results."""
    # Plot the relative 3D position
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Make  ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xell = axes[0] * np.outer(np.cos(u), np.sin(v))
    yell = axes[1] * np.outer(np.sin(u), np.sin(v))
    zell = axes[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xell/1e3, yell/1e3, zell/1e3, rstride=4, cstride=4, color=(105/255,105/255,105/255))

    ax.plot(rTruth[:,0]/1e3, rTruth[:,1]/1e3, rTruth[:,2]/1e3, 'b', label='Truth')
    ax.plot(r1[:,0]/1e3, r1[:,1]/1e3, r1[:,2]/1e3, label='HOU')
    ax.plot(r2[:,0]/1e3, r2[:,1]/1e3, r2[:,2]/1e3, label='UKF')
    ax.plot(r3[:,0]/1e3, r3[:,1]/1e3, r3[:,2]/1e3, label='Kep.')

    ax.set_xlabel('${}^{A}r_{x}$ [km]')
    ax.set_ylabel('${}^{A}r_{y}$ [km]')
    ax.set_zlabel('${}^{A}r_{z}$ [km]')
    ax.set_title('Small body fixed frame')

    ax.legend(loc='upper left')

def plot_comparisonCS2ndOrder(trainingUKFData, scenarioIntUKFOut, CTruth, STruth, figName=[], smallbody=[]):
    """Plot the global gravity results."""
    # Extract data
    tIntUKF = scenarioIntUKFOut.time
    CEstIntUKF = scenarioIntUKFOut.CEst
    SEstIntUKF = scenarioIntUKFOut.SEst

    tUKF = trainingUKFData.tCSmat
    CEstUKF = trainingUKFData.Cmat
    SEstUKF = trainingUKFData.Smat
    tUKF = np.hstack((0,tUKF))
    CEstUKF = np.dstack((np.zeros((5,5)),CEstUKF))
    SEstUKF = np.dstack((np.zeros((5,5)),SEstUKF))

    tTruth = [tIntUKF[0]/(3600*24),tIntUKF[-1]/(3600*24)]

    plt.gcf()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,2,0], color='royalblue', label='$C_{20}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,2,1], color='red', label='$C_{21}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,2,2], color='gold', label='$C_{22}$')

    ax[0].plot(tTruth, [CTruth[2,0],CTruth[2,0]], color='royalblue', linestyle='dotted')
    ax[0].plot(tTruth, [CTruth[2,1],CTruth[2,1]], color='red', linestyle='dotted')
    ax[0].plot(tTruth, [CTruth[2,2],CTruth[2,2]], color='gold', linestyle='dotted')

    ax[0].step(tUKF/(3600*24), CEstUKF[2,0,:], color='royalblue',marker="x",linestyle='dashed')
    ax[0].step(tUKF/(3600*24), CEstUKF[2,1,:], color='red',marker="x",linestyle='dashed')
    ax[0].step(tUKF/(3600*24), CEstUKF[2,2,:], color='gold',marker="x",linestyle='dashed')

    ax[0].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[0].set_xlabel('Time [days]')
    ax[0].set_ylabel('$C_{2i}$ [-]')
    ax[0].legend(loc='lower left')

    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,2,1], color='royalblue', label='$S_{21}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,2,2], color='red', label='$S_{22}$')

    ax[1].plot(tTruth, [STruth[2,1],STruth[2,1]], color='royalblue',linestyle='dotted')
    ax[1].plot(tTruth, [STruth[2,2],STruth[2,2]], color='red',linestyle='dotted')

    ax[1].step((tUKF-tUKF[0])/(3600*24), SEstUKF[2,1,:], color='royalblue',marker="x",linestyle='dashed')
    ax[1].step((tUKF-tUKF[0])/(3600*24), SEstUKF[2,2,:], color='red',marker="x",linestyle='dashed')

    ax[1].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('$S_{2i}$ [-]')
    ax[1].legend(loc='lower left')

    plt.tight_layout()

    if figName:
        figName = 'Plots/' + smallbody + '_' + figName + 'CS2order' + '.eps'
        plt.savefig(figName, format='eps')


def plot_comparisonCS3rdOrder(trainingUKFData, scenarioIntUKFOut, CTruth, STruth, figName=[], smallbody=[]):
    """Plot the global gravity results."""
    # Extract data
    tIntUKF = scenarioIntUKFOut.time
    CEstIntUKF = scenarioIntUKFOut.CEst
    SEstIntUKF = scenarioIntUKFOut.SEst

    tUKF = trainingUKFData.tCSmat
    CEstUKF = trainingUKFData.Cmat
    SEstUKF = trainingUKFData.Smat
    tUKF = np.hstack((0,tUKF))
    CEstUKF = np.dstack((np.zeros((5,5)),CEstUKF))
    SEstUKF = np.dstack((np.zeros((5,5)),SEstUKF))

    tTruth = [tIntUKF[0] / (3600 * 24), tIntUKF[-1] / (3600 * 24)]

    plt.gcf()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,3,0], color='royalblue', label='$C_{30}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,3,1], color='red', label='$C_{31}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,3,2], color='gold', label='$C_{32}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,3,3], color='purple', label='$C_{33}$')

    ax[0].plot(tTruth, [CTruth[3,0],CTruth[3,0]], color='royalblue',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[3,1],CTruth[3,1]], color='red',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[3,2],CTruth[3,2]], color='gold',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[3,3],CTruth[3,3]], color='purple',linestyle='solid')

    ax[0].step(tUKF/(3600*24), CEstUKF[3,0,:], color='royalblue', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[3,1,:], color='red', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[3,2,:], color='gold', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[3,3,:], color='purple', linestyle='dotted', marker="x")

    ax[0].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[0].set_xlabel('Time [days]')
    ax[0].set_ylabel('$C_{3i}$ [-]')
    ax[0].legend(loc='lower left')

    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,3,1], color='royalblue', label='$S_{31}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,3,2], color='red', label='$S_{32}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,3,3], color='gold', label='$S_{33}$')

    ax[1].step(tUKF/(3600*24), SEstUKF[3,1,:], color='royalblue', linestyle='dotted', marker="x")
    ax[1].step(tUKF/(3600*24), SEstUKF[3,2,:], color='red', linestyle='dotted', marker="x")
    ax[1].step(tUKF/(3600*24), SEstUKF[3,3,:], color='gold', linestyle='dotted', marker="x")

    ax[1].plot(tTruth, [STruth[3,1],STruth[3,1]], color='royalblue',linestyle='solid')
    ax[1].plot(tTruth, [STruth[3,2],STruth[3,2]], color='red',linestyle='solid')
    ax[1].plot(tTruth, [STruth[3,3],STruth[3,3]], color='gold',linestyle='solid')

    ax[1].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('$S_{3i}$ [-]')
    ax[1].legend(loc='lower left')

    plt.tight_layout()

    if figName:
        figName = 'Plots/' + smallbody + '_' + figName + 'CS3order' + '.eps'
        plt.savefig(figName, format='eps')


def plot_comparisonCS4thOrder(trainingUKFData, scenarioIntUKFOut, CTruth, STruth, figName=[], smallbody=[]):
    """Plot the global gravity results."""
    # Extract data
    tIntUKF = scenarioIntUKFOut.time
    CEstIntUKF = scenarioIntUKFOut.CEst
    SEstIntUKF = scenarioIntUKFOut.SEst

    tUKF = trainingUKFData.tCSmat
    CEstUKF = trainingUKFData.Cmat
    SEstUKF = trainingUKFData.Smat
    tUKF = np.hstack((0,tUKF))
    CEstUKF = np.dstack((np.zeros((5,5)),CEstUKF))
    SEstUKF = np.dstack((np.zeros((5,5)),SEstUKF))

    tTruth = [tIntUKF[0] / (3600 * 24), tIntUKF[-1] / (3600 * 24)]

    plt.gcf()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,4,0], color='royalblue', label='$C_{40}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,4,1], color='red', label='$C_{41}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,4,2], color='gold', label='$C_{42}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,4,3], color='purple', label='$C_{43}$')
    ax[0].plot(tIntUKF/(3600*24), CEstIntUKF[:,4,4], color='limegreen', label='$C_{44}$')

    ax[0].plot(tTruth, [CTruth[4,0],CTruth[4,0]], color='royalblue',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[4,1],CTruth[4,1]], color='red',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[4,2],CTruth[4,2]], color='gold',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[4,3],CTruth[4,3]], color='purple',linestyle='solid')
    ax[0].plot(tTruth, [CTruth[4,4],CTruth[4,4]], color='limegreen',linestyle='solid')

    ax[0].step(tUKF/(3600*24), CEstUKF[4,0,:], color='royalblue', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[4,1,:], color='red', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[4,2,:], color='gold', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[4,3,:], color='purple', linestyle='dotted', marker="x")
    ax[0].step(tUKF/(3600*24), CEstUKF[4,4,:], color='limegreen', linestyle='dotted', marker="x")

    ax[0].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[0].set_xlabel('Time [days]')
    ax[0].set_ylabel('$C_{4i}$ [-]')
    ax[0].legend(loc='lower left')

    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,4,1], color='royalblue', label='$S_{41}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,4,2], color='red', label='$S_{42}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,4,3], color='gold', label='$S_{43}$')
    ax[1].plot(tIntUKF/(3600*24), SEstIntUKF[:,4,4], color='purple', label='$S_{44}$')

    ax[1].plot(tTruth, [STruth[4,1],STruth[4,1]], color='royalblue',linestyle='solid')
    ax[1].plot(tTruth, [STruth[4,2],STruth[4,2]], color='red',linestyle='solid')
    ax[1].plot(tTruth, [STruth[4,3],STruth[4,3]], color='gold',linestyle='solid')
    ax[1].plot(tTruth, [STruth[4,4],STruth[4,4]], color='purple',linestyle='solid')

    ax[1].step(tUKF/(3600*24), SEstUKF[4,1,:], color='royalblue', linestyle='dotted', marker="x")
    ax[1].step(tUKF/(3600*24), SEstUKF[4,2,:], color='red', linestyle='dotted', marker="x")
    ax[1].step(tUKF/(3600*24), SEstUKF[4,3,:], color='gold', linestyle='dotted', marker="x")
    ax[1].step(tUKF/(3600*24), SEstUKF[4,4,:], color='purple', linestyle='dotted', marker="x")

    ax[1].set_xlim([tIntUKF[0]/(24*3600), tIntUKF[-1]/(24*3600)])
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('$S_{4i}$ [-]')
    ax[1].legend(loc='lower left')

    plt.tight_layout()

    if figName:
        figName = 'Plots/' + smallbody + '_' + figName + 'CS4order' + '.eps'
        plt.savefig(figName, format='eps')


def plot_errCS(trainingUKFData, scenarioIntUKFOut, CTruth, STruth, figName=[], smallbody=[]):
    CSErr_UKF = np.zeros(3)
    CSErr_IntUKF = np.zeros(3)

    Cmat_UKF = trainingUKFData.Cmat[:,:,-1]
    Smat_UKF = trainingUKFData.Smat[:,:,-1]

    Cmat_IntUKF = scenarioIntUKFOut.CEst[-1,:,:]
    Smat_IntUKF = scenarioIntUKFOut.SEst[-1,:,:]

    deg = len(Cmat_UKF)

    for ii in range(2,deg):
        cont = 0
        for jj in range(0,ii+1):
            CSErr_UKF[ii-2] += (Cmat_UKF[ii,jj]-CTruth[ii,jj])**2
            CSErr_IntUKF[ii-2] += (Cmat_IntUKF[ii,jj] - CTruth[ii,jj]) ** 2
            cont += 1
            if jj > 1:
                CSErr_UKF[ii-2] += (Smat_UKF[ii,jj] - STruth[ii,jj])**2
                CSErr_IntUKF[ii-2] += (Smat_IntUKF[ii,jj] - STruth[ii,jj]) ** 2
                cont += 1

        CSErr_UKF[ii-2] = np.sqrt(CSErr_UKF[ii-2]/cont)
        CSErr_IntUKF[ii-2] = np.sqrt(CSErr_IntUKF[ii-2]/cont)

    """Plot the spherical harmonics RMS."""
    plt.gcf()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([2,3,4],CSErr_UKF, 'b', label='Least-squares', linestyle='dashed', marker='.',markersize=12)
    ax.plot([2,3,4],CSErr_IntUKF, color='orange', label='Extended state', linestyle='dashed', marker='.',markersize=12)
    plt.xticks([2,3,4])
    ax.set_xlabel('Order [-]')
    ax.set_ylabel('RMS [-]')

    ax.legend(loc='upper left')
    plt.tight_layout
    if figName:
        figName = 'Plots/' + smallbody + '_' + figName + 'CSrms' + '.eps'
        plt.savefig(figName, format='eps')

def plot_comparisonCPUtime(scenario1Out, scenario2Out):
    """Plot bar diagram with cpu times."""
    # Sum the iterations for each segment
    tcpuTotalFilter1 = np.sum(scenario1Out.tcpuTotalFilter,axis=1)
    tcpuTrain1 = np.sum(scenario1Out.tcpuTrain, axis=1)
    tcpuTotalFilter2 = np.sum(scenario2Out.tcpuTotalFilter,axis=1)
    tcpuTrain2 = np.sum(scenario2Out.tcpuTrain, axis=1)

    nSegments1 = len(tcpuTotalFilter1)
    nSegments2 = len(tcpuTotalFilter2)

    plt.gcf()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].bar(np.linspace(1,nSegments1,nSegments1),tcpuTotalFilter1)
    ax[0].bar(np.linspace(1,nSegments1,nSegments1),tcpuTrain1, bottom=tcpuTotalFilter1)
    ax[0].set_ylabel('Computational time [s]')
    ax[1].bar(np.linspace(1,nSegments2,nSegments2),tcpuTotalFilter2)
    ax[1].bar(np.linspace(1,nSegments2,nSegments2),tcpuTrain2, bottom=tcpuTotalFilter2)
    ax[1].set_ylabel('Segments [-]')
    ax[1].set_ylabel('Computational time [s]')



