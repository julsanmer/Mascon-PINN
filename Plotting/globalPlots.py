import matplotlib.pyplot as plt
import numpy as np

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

def plot_UKFRMSseg(rRMS, vRMS, aRMS, labels):
    _, nSegments, nSim = rRMS.shape
    rRMS = rRMS.reshape((nSegments,nSim))
    vRMS = vRMS.reshape((nSegments,nSim))
    aRMS = aRMS.reshape((nSegments,nSim))

    """Plot the RMS results."""
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    for ii in range(nSim):
        ax[0].plot(rRMS[:,ii], label=labels[ii],marker='o', linestyle='--')
        ax[1].plot(vRMS[:,ii],marker='o', linestyle='--')
        ax[2].plot(aRMS[:,ii]*1e3,marker='o', linestyle='--')

    plt.xlabel('Segments [-]')

    ax[0].set_ylabel('Position RMS [m]')
    ax[1].set_ylabel('Velocity RMS [m/s]')
    ax[2].set_ylabel('Acceleration RMS [mm/s$^2$]')

    ax[0].legend()


def plot_CSRMSseg(CSRMS, labels):
    _, maxDeg, nSim = CSRMS.shape

    """Plot the RMS results."""
    plt.gcf()
    fig, ax = plt.subplots(1, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    for ii in range(nSim):
        ax.plot(CSRMS[-1,:,ii], label=labels[ii],marker='o', linestyle='--')

    plt.xlabel('Segments [-]')
    ax.legend()


def plot_aErr3Dpdf(pdfbins, aErr3Dpdf, legendlabels):
    _, nBins, nSim = pdfbins.shape
    pdfbins = pdfbins.reshape((nBins,nSim))
    _, nBins, nSim = aErr3Dpdf.shape
    aErr3Dpdf = aErr3Dpdf.reshape((nBins,nSim))

    """Plot pdf of acceleration error."""
    plt.gcf()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ii in range(nSim):
        ax.plot(pdfbins[0:-1,ii],aErr3Dpdf[:,ii], marker='o', linestyle='--', label=legendlabels[ii])

    plt.xlabel('Gravity acceleration error [%]')
    plt.ylabel('Pdf [-]')

    ax.legend(loc='upper right')


def plot_UKFRMS(degSpherHarm, rRMS, vRMS, aRMS):
    if isinstance(aRMS,np.ndarray):
        nplots = 3
    else:
        nplots = 2

    """Plot the relative velocity result."""
    plt.gcf()
    fig, ax = plt.subplots(nplots, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(degSpherHarm, rRMS, 'b', marker='o', linestyle='--')
    ax[0].ticklabel_format(useOffset=False)
    ax[1].plot(degSpherHarm, vRMS, 'b', marker='o', linestyle='--')
    ax[1].ticklabel_format(useOffset=False)
    if isinstance(aRMS,np.ndarray):
        ax[2].plot(degSpherHarm, aRMS*1e3, 'b', marker='o', linestyle='--')
        ax[2].ticklabel_format(useOffset=False)

    plt.xlabel('Gravity degree [-]')
    plt.title('Filter RMS')

    ax[0].set_ylabel('Position [m]')
    ax[1].set_ylabel('Velocity [m/s]')
    if isinstance(aRMS,np.ndarray):
        ax[2].set_ylabel('Acceleration [mm/s$^2$]')


def plot_CSRMS(degSpherHarm, CSRMS1, CSRMS2, legendlabels, titles):
    _, degmax, nSim = CSRMS1.shape
    CSRMS1 = CSRMS1.reshape((degmax,nSim))

    """Plot RMS of gravity parameters."""
    plt.gcf()

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1, 1, 1)

    for ii in range(nSim):
        ax.plot(np.linspace(2,degSpherHarm[ii],degSpherHarm[ii]-1),CSRMS1[0:degSpherHarm[ii]-1,ii],
                marker='o', linestyle='-',label=legendlabels[ii], zorder=10, clip_on=False)
    if isinstance(CSRMS2,np.ndarray):
        _, degmax, nSim = CSRMS2.shape
        CSRMS2 = CSRMS2.reshape((degmax, nSim))
        plt.gca().set_prop_cycle(None)
        for ii in range(nSim):
            ax.plot(np.linspace(2, degSpherHarm[ii], degSpherHarm[ii] - 1), CSRMS2[0:degSpherHarm[ii]-1, ii],
                    marker='X', linestyle='--', zorder=10, clip_on=False)

    ax.plot([2,degSpherHarm[-1]],[1,1],'k',linewidth=2.5)
    ax.set_xlim(2, degSpherHarm[-1])

    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel('Gravity degree [-]',fontsize = 12)
    plt.ylabel('Gravity normalized error [-]',fontsize=12)
    ax.set_yscale('log')
    plt.grid()

    if isinstance(CSRMS2,np.ndarray):
        lines = plt.gca().get_lines()
        legend1 = plt.legend([lines[ii - 2] for ii in degSpherHarm], [lines[ii - 2].get_label() for ii in degSpherHarm],
                             loc=4, fontsize=11)
        legend2 = plt.legend([lines[ii] for ii in [0, degSpherHarm[-1]-1]], titles, loc=0, fontsize=11)
        plt.gca().add_artist(legend1)
    plt.savefig('Plots/CSErr.eps', format='eps')


def plot_aErr3DInterval(degNav, rInterval, aErr3DInterval, legendlabels):
    """Plot pdf of acceleration error."""
    _, rbins, nSim = aErr3DInterval.shape
    #rInterval = rInterval.reshape((rbins,nSim))
    aErr3DInterval = aErr3DInterval.reshape((rbins,nSim))

    x_ticks_labels = []
    x = np.linspace(1,rbins,rbins)
    for ii in range(rbins):
        if ii == 0:
            r0str = 'Surf.'
        else:
            r0str = str(rInterval[ii,0,0]/1e3)
        rfstr = str(rInterval[ii,1,0]/1e3)
        x_ticks_labels.append(r0str + '-' + rfstr)

    plt.gcf()

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1, 1, 1)

    for ii in range(nSim):
        ax.plot(x,aErr3DInterval[:,ii]*1e2,marker='o', linestyle='-',label=legendlabels[ii], zorder=10, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks_labels)

    ax.set_yscale('log')

    plt.xlabel('Radius range [km]')
    plt.ylabel('Gravity acceleration error [%]')

    ax.grid()

    ax.legend(loc='upper right')


def plot_tcpu(degSpherHarm, tcpuUKF, tcpuLSQ, tcpuIntUKF):
    """Plot bar diagram with cpu times."""
    # Sum the iterations for each segment
    width = 0.35
    if isinstance(tcpuIntUKF,np.ndarray):
        degSpherHarm1 = [x - width/2 for x in degSpherHarm]
        degSpherHarm2 = [x + width/2 for x in degSpherHarm]
    else:
        degSpherHarm1 = degSpherHarm

    plt.gcf()
    fig = plt.figure(figsize=(6,5))
    ax = plt.subplot()
    ax.bar(degSpherHarm1, tcpuUKF, width=width, label='Dynamic compensation UKF')
    ax.bar(degSpherHarm1, tcpuLSQ, bottom=tcpuUKF, width=width, label='Least-squares')
    if isinstance(tcpuIntUKF,np.ndarray):
        ax.bar(degSpherHarm2, tcpuIntUKF, color='g', width=width, label='Extended state UKF')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Gravity degree [-]',fontsize = 12)
    ax.set_ylabel('Computational time [s]',fontsize = 12)
    plt.legend(loc='upper left',fontsize=11)
    plt.grid()
    plt.savefig('Plots/tcpu.eps', format='eps')


def plot_tcpu2(degSpherHarm, tcpuTotal1, tcpuTotal2,labels,iters):
    """Plot bar diagram with cpu times."""
    # Sum the iterations for each segment
    width = 0.35
    degSpherHarm1 = [x - width/2 for x in degSpherHarm]
    degSpherHarm2 = [x + width/2 for x in degSpherHarm]

    plt.gcf()
    fig = plt.figure(figsize=(6,5))
    ax1 = plt.subplot()
    ax1.bar(degSpherHarm1, tcpuTotal1, width=width, label=labels[0])
    ax1.bar(degSpherHarm2, tcpuTotal2, width=width, label=labels[1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('Gravity degree [-]',fontsize = 12)
    ax1.set_ylabel('Computational time [s]',fontsize = 12)
    plt.legend(loc='upper left',fontsize=11)
    plt.grid()

    ax2 = ax1.twinx()
    ax2.plot(degSpherHarm2,iters,linestyle='--', marker='o', color='k', label='Iterations')
    ax2.set_ylabel('Iterations [-]',fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    lines = plt.gca().get_lines()
    legend1 = plt.legend(lines, ['Iterations'], fontsize=11, loc=4)

    plt.savefig('Plots/tcpu.eps', format='eps')
