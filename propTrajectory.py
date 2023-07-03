import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot

import numpy as np
import os
import pickle

from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# import simulation related support
from Basilisk.simulation import ephemerisConverter, planetEphemeris, spacecraft, radiationPressure
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                RigidBodyKinematics, simIncludeGravBody, unitTestSupport)
from Basilisk.architecture import sim_model
from Basilisk.simulation.gravityEffector import loadPolyFromFileToList

# always import the Basilisk messaging support
from Basilisk.architecture import messaging


def propSpacecraft(grav, C, S, deg, posM, muM, scenarioType):
    # Set parameters
    dtSampling = 60
    dtIntegration = 1

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    # Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)

    # Create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and specify the integration update time
    samplingTime = macros.sec2nano(dtSampling)
    simulationTimeStep = macros.sec2nano(dtIntegration)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # setup celestial object ephemeris module
    gravBodyEphem = planetEphemeris.PlanetEphemeris()
    gravBodyEphem.ModelTag = 'erosEphemeris'
    gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(["eros"]))

    # specify small body o.e. and rotational state January 21st, 2022
    # https://ssd.jpl.nasa.gov/horizons.cgi#results
    oeAsteroid = planetEphemeris.ClassicElementsMsgPayload()
    oeAsteroid.a = 2.3612 * orbitalMotion.AU * 1000  # meters
    oeAsteroid.e = 0.08823
    oeAsteroid.i = 7.1417 * macros.D2R
    oeAsteroid.Omega = 103.8 * macros.D2R
    oeAsteroid.omega = 151.1 * macros.D2R
    oeAsteroid.f = 7.0315 * macros.D2R
    r_AN_N, v_AN_N = orbitalMotion.elem2rv(orbitalMotion.MU_SUN * (1000.**3), oeAsteroid)

    # the rotational state would be prescribed to
    AR = 11.369 * macros.D2R
    dec = 17.227*macros.D2R  # 42.23 * macros.D2R
    lst0 = 0*macros.D2R
    rotPeriod = 5.27 * 3600
    gravBodyEphem.planetElements = planetEphemeris.classicElementVector([oeAsteroid])
    gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([AR])
    gravBodyEphem.declination = planetEphemeris.DoubleVector([dec])
    gravBodyEphem.lst0 = planetEphemeris.DoubleVector([lst0])
    gravBodyEphem.rotRate = planetEphemeris.DoubleVector([360 * macros.D2R / rotPeriod])

    # Create gravity model
    mu = 4.46275472004*1e5
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createCustomGravObject('eros', 4.46275472004*1e5, radEquator=16*1e3)
    if grav == 'spherharm':
        planet.useSphericalHarmParams = True
        #print(planet.spherHarm.muBody)
        simIncludeGravBody.loadGravFromFile('/Users/julio/basilisk/supportData/LocalGravData/EROS15A.txt', planet.spherHarm, int(deg))
        planet.spherHarm.cBar = sim_model.MultiArray(C.tolist())
        planet.spherHarm.sBar = sim_model.MultiArray(S.tolist())
        #print(planet.spherHarm.cBar)
    elif grav == 'poly':
        planet.usePolyhedral = True
        simIncludeGravBody.loadPolyFromFile('/Users/julio/basilisk/supportData/LocalGravData/eros007790.tab', planet.poly)
    elif grav == 'mascon':
        planet.useMascon = True
        planet.mascon.posM = posM
        planet.mascon.muM = muM
    planet.planetBodyInMsg.subscribeTo(gravBodyEphem.planetOutMsgs[0])

    gravFactory.createSun()

    # Create a sun spice message, zero it out, required by srp
    sunPlanetStateMsgData = messaging.SpicePlanetStateMsgPayload()
    sunPlanetStateMsg = messaging.SpicePlanetStateMsg()
    sunPlanetStateMsg.write(sunPlanetStateMsgData)

    # Create an SRP model
    srp = radiationPressure.RadiationPressure()  # default model is the SRP_CANNONBALL_MODEL
    srp.area = 1.1  # m^3
    srp.coefficientReflection = 1.2
    srp.sunEphmInMsg.subscribeTo(sunPlanetStateMsg)

    # create an ephemeris converter
    ephemConverter = ephemerisConverter.EphemerisConverter()
    ephemConverter.ModelTag = "ephemConverter"
    ephemConverter.addSpiceInputMsg(gravBodyEphem.planetOutMsgs[0])

    # The dynamics simulation is setup using a Spacecraft() module.
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # Finally, the gravitational body must be connected to the spacecraft object.  This is done with
    scObject.hub.mHub = 750.0
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))
    scObject.addDynamicEffector(srp)

    # Setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    if scenarioType == 'lowOrbit':
        oe.a = 20*1e3
        oe.i = 90*np.pi/180
    elif scenarioType == 'medOrbit':
        oe.a = 34*1e3
        oe.i = 45*np.pi/180
    elif scenarioType == 'highOrbit':
        oe.a = 44*1e3
        oe.i = 170*np.pi/180
    oe.e = 0.001
    oe.Omega = 48.2*np.pi/180
    oe.omega = 347.8*np.pi/180
    oe.f = 85.3*np.pi/180

    dcm_AN = RigidBodyKinematics.euler3232C([AR, np.pi/2-dec, lst0])
    r_CA, v_CA = orbitalMotion.elem2rv(mu, oe)
    r_CA_N = dcm_AN.transpose().dot(r_CA)
    v_CA_N = dcm_AN.transpose().dot(v_CA)

    # Set the ICs for SC position and velocity in inertial frame
    r_CN_N = r_CA_N
    v_CN_N = v_CA_N

    # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
    scObject.hub.r_CN_NInit = r_AN_N + r_CN_N
    scObject.hub.v_CN_NInit = v_AN_N + v_CN_N

    # Set the simulation time
    n = np.sqrt(mu / oe.a / oe.a / oe.a)
    Torb = 2.*np.pi/n
    nOrbits = 1
    #simulationTime = macros.sec2nano(nOrbits*Torb)
    simulationTime = macros.sec2nano(12*3600)


    scSim.AddModelToTask(simTaskName, gravBodyEphem, ModelPriority=100)
    scSim.AddModelToTask(simTaskName, ephemConverter, ModelPriority=99)
    scSim.AddModelToTask(simTaskName, scObject, ModelPriority=98)
    scSim.AddModelToTask(simTaskName, srp, ModelPriority=98)

    # Set number of data points
    scRec = scObject.scStateOutMsg.recorder(samplingTime)
    astRec = ephemConverter.ephemOutMsgs[0].recorder(samplingTime)

    scSim.AddModelToTask(simTaskName, scRec)
    scSim.AddModelToTask(simTaskName, astRec)

    # Set simulation
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # Retrieve data from the simulation
    time = np.array(scRec.times() * macros.NANO2SEC)
    pos_N = np.array(scRec.r_BN_N - astRec.r_BdyZero_N)
    vel_N = np.array(scRec.v_BN_N - astRec.v_BdyZero_N)

    # Retrieve small body rotational state
    sigma_AN = astRec.sigma_BN
    omega_AN_A = astRec.omega_BN_B

    # Preallocate variables
    pos_A = np.zeros((len(time),3))
    vel_A = np.zeros((len(time),3))

    # Loop through times
    for ii in range(len(time)):
        # obtain rotation matrix
        dcm_AN = RigidBodyKinematics.MRP2C(sigma_AN[ii,0:3])

        # rotate position and velocity
        pos_A[ii,0:3] = dcm_AN.dot(pos_N[ii,0:3])
        vel_A[ii,0:3] = dcm_AN.dot(vel_N[ii,0:3]) - np.cross(omega_AN_A[ii,0:3], pos_A[ii,0:3])

    return time, pos_A


if __name__ == "__main__":
    propagateFlag = True
    plotFlag = False
    scenarioType = 'medOrbit'
    pathFile = 'Results/Propagation/'

    if propagateFlag:
        time, posTruth_A = propSpacecraft('poly', [], [], [], [], [],scenarioType)
        time, pos = propSpacecraft('', [], [], [], [], [],scenarioType)
        dpos = posTruth_A - pos
        RMS0 = np.zeros(len(time))
        for jj in range(len(time)):
            RMS0[jj] = np.sqrt(np.sum(dpos[0:jj, 0:3]**2) / (jj+1))

        for ii in range(7):
            filenameSpherharm = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/spherharm' + str(ii+2) + 'th.pck'
            navParams, navOutputs = pickle.load(open(filenameSpherharm, "rb"))
            CEst = navOutputs.nav.CEst[-1,:,:]
            SEst = navOutputs.nav.SEst[-1,:,:]
            deg = navParams.navfilter.degSmallBody

            # Propagate
            time, pos = propSpacecraft('spherharm', CEst, SEst, deg, [], [], scenarioType)

            # Compute RMS
            dpos = posTruth_A - pos
            RMS = np.zeros(len(time))
            for jj in range(len(time)):
                RMS[jj] = np.sqrt(np.sum(dpos[0:jj,0:3]**2)/(jj+1))
            print(RMS[-1])
            dataFile = pathFile + scenarioType + '/SH' + str(ii+2) + 'th.pck'
            with open(dataFile, "wb") as f:
                pickle.dump([time, posTruth_A, RMS], f)

        for ii in range(7):
            filenameSpherharm = 'Results/eros_poly_intUKF_10orbits0iter_shadow/spherharm' + str(ii+2) + 'th.pck'
            navParams, navOutputs = pickle.load(open(filenameSpherharm, "rb"))
            CEst = navOutputs.nav.CEst[-1,:,:]
            SEst = navOutputs.nav.SEst[-1,:,:]
            deg = navParams.navfilter.degSmallBody

            # Propagate
            time, pos = propSpacecraft('spherharm', CEst, SEst, deg, [], [], scenarioType)
            dpos = posTruth_A - pos
            RMS = np.zeros(len(time))
            for jj in range(len(time)):
                RMS[jj] = np.sqrt(np.sum(dpos[0:jj,0:3]**2)/(jj+1))
            print(RMS[-1])
            dataFile = pathFile + scenarioType + '/SHint' + str(ii+2) + 'th.pck'
            with open(dataFile, "wb") as f:
                pickle.dump([time, posTruth_A, RMS], f)

        nRand = 1
        nM = 160
        for ii in range(nRand):
            # Load filename
            filenameMascon = 'Results/eros_poly_lsqUKF_10orbits0iter_shadow/mascons' + str(nM) + 'sqrLAM1E-1WinitV2Test/rand' + str(ii) + '.pck'
            navParams, navOutputs = pickle.load(open(filenameMascon, "rb"))
            posM = navOutputs.nav.posMEst
            muM = navOutputs.nav.muMEst

            # Propagate
            time, pos = propSpacecraft('mascon', CEst, SEst, deg, posM, muM, scenarioType)
            dpos = posTruth_A - pos
            RMS = np.zeros(len(time))
            for jj in range(len(time)):
                RMS[jj] = np.sqrt(np.sum(dpos[0:jj,0:3]**2)/(jj+1))
            dataFile = pathFile + scenarioType + '/M' + str(nM) + '/rand' + str(ii) + '.pck'
            #with open(dataFile, "wb") as f:
            #    pickle.dump([time, posTruth_A, RMS], f)
            #print(ii)
        print(RMS[-1])

    if plotFlag:
        time, posTruth_A, RMS_M, RMS_SH, RMS0 = pickle.load(open(dataFile, "rb"))
        nRand = len(RMS_M)
        _, nRand = RMS_M.shape

        # Compute estimation
        fig1 = plt.figure(figsize=(6,5))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(time / (3600*24), RMS0, color='black', linestyle='-.', zorder=10)
        for ii in range(7):
            ax1.plot(time/(3600*24),RMS_SH[:,ii],color='red',linestyle='--',zorder=10)
        for ii in range(nRand):
            ax1.plot(time/(3600*24),RMS_M[:,ii],color='blue',linewidth=0.1)
        #ax1.plot(time/(3600*24),RMSSH,color='red')

        # Get polyhedron and landmarks
        color_smallbody = [105/255, 105/255, 105/255]
        polyFilename = '/Users/julio/basilisk/supportData/LocalGravData/eros007790.tab'
        vertList, faceList, nVertex, nFacet = loadPolyFromFileToList(polyFilename)
        xyzPoly = np.array(vertList)
        idxPoly = np.array(faceList)

        fig2 = plt.figure(figsize=(12,6))
        ax2 = fig2.add_subplot(1, 1, 1, projection='3d')

        ax2.plot_trisurf(xyzPoly[:,0]/1e3, xyzPoly[:,1]/1e3, xyzPoly[:,2]/1e3, triangles=idxPoly-1, color=color_smallbody, zorder=0)
        ax2.plot(posTruth_A[:,0]/1e3, posTruth_A[:,1]/1e3, posTruth_A[:,2]/1e3, 'b', zorder=20, linewidth=0.5)
        ax2.tick_params(axis='both')
        ax2.set_facecolor('white')

        plt.show()