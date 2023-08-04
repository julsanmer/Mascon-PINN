import numpy as np
import os
import inspect

from Basilisk.simulation import planetEphemeris
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros as mc

from Basilisk.ExternalModules import masconFit

# Import master classes: simulation base class and scenario base class
from BSK.BSK_masters import BSKSim, BSKScenario
import BSK.Dynamics.BSK_Dynamics as BSK_Dynamics

# Get current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


# Create your own scenario child class
class ScenarioAsteroid(BSKSim, BSKScenario):
    def __init__(self, parameters):
        super(ScenarioAsteroid, self).__init__()
        self.name = 'ScenarioAsteroid'

        # Initialize class variables
        self.asteroidEphemRec = None
        self.asteroidTruthRec = None
        self.asteroidMeasRec = None
        self.scTruthRec = None
        self.scCamMeasRec = None
        self.scSimpleMeasRec = None
        self.scNavRec = None

        # Initialize DMC-UKF state
        self.x0 = np.zeros(9)

        # Set dynamics subclass
        self.set_DynModel(BSK_Dynamics, parameters)

        # Set dynamics initial condition and loggers
        self.configure_dyn_initial_conditions(parameters)
        self.log_outputs(parameters.save_rate)

    # This subclass sets initial conditions for asteroid and spacecraft
    def configure_dyn_initial_conditions(self, parameters):
        # Get dynamics model
        dyn_model = self.get_DynModel()

        # Set asteroid ephemeris
        oe_asteroid = planetEphemeris.ClassicElementsMsgPayload()
        oe_asteroid.a = parameters.asteroid.a
        oe_asteroid.e = parameters.asteroid.ecc
        oe_asteroid.i = parameters.asteroid.inc
        oe_asteroid.Omega = parameters.asteroid.RAAN
        oe_asteroid.omega = parameters.asteroid.omega
        oe_asteroid.f = parameters.asteroid.f
        ra = parameters.asteroid.ra
        dec = parameters.asteroid.dec
        lst0 = parameters.asteroid.lst0
        rot_period = parameters.asteroid.rot_period
        dyn_model.gravBodyEphem.planetElements = planetEphemeris.classicElementVector([oe_asteroid])
        dyn_model.gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([ra])
        dyn_model.gravBodyEphem.declination = planetEphemeris.DoubleVector([dec])
        dyn_model.gravBodyEphem.lst0 = planetEphemeris.DoubleVector([lst0])
        dyn_model.gravBodyEphem.rotRate = planetEphemeris.DoubleVector([360*mc.D2R / rot_period])

        # Compute asteroid dcm and angular velocity w.r.t. inertial
        dcm_AN = rbk.euler3232C([ra, np.pi/2 - dec, lst0])
        omega_AN_A = np.array([0, 0, 360*mc.D2R/rot_period])
        r_AN_N, v_AN_N = orbitalMotion.elem2rv(orbitalMotion.MU_SUN*(1000.**3), oe_asteroid)

        # Retrieve gravity parameter
        mu_asteroid = dyn_model.gravFactory.gravBodies[parameters.asteroid.name].mu

        # Set spacecraft initial condition
        oe_spacecraft = orbitalMotion.ClassicElements()
        oe_spacecraft.a = parameters.spacecraft.a
        oe_spacecraft.e = parameters.spacecraft.ecc
        oe_spacecraft.i = parameters.spacecraft.inc
        oe_spacecraft.Omega = parameters.spacecraft.RAAN
        oe_spacecraft.omega = parameters.spacecraft.omega
        oe_spacecraft.f = parameters.spacecraft.f
        r_CA, v_CA = orbitalMotion.elem2rv(mu_asteroid, oe_spacecraft)

        # Set DMC-UKF initial estimate
        parameters.dmcukf.x_k[0:3] = r_CA
        parameters.dmcukf.x_k[3:6] = v_CA - np.cross(omega_AN_A, r_CA)

        # Rotate spacecraft initial condition to inertial coordinates
        r_CA_N = dcm_AN.transpose().dot(r_CA)
        v_CA_N = dcm_AN.transpose().dot(v_CA)

        # Set spacecraft initial condition in BSK
        if parameters.flag_sun:
            r_CN_N = r_CA_N + r_AN_N
            v_CN_N = v_CA_N + v_AN_N
        else:
            r_CN_N = r_CA_N
            v_CN_N = v_CA_N
        dyn_model.scObject.hub.r_CN_NInit = r_CN_N
        dyn_model.scObject.hub.v_CN_NInit = v_CN_N

    # This subclass adds logging tasks to basilisk
    def log_outputs(self, dt_sampling):
        # Get dynamics model
        dyn_model = self.get_DynModel()

        # Set recorders
        self.asteroidEphemRec = dyn_model.gravBodyEphem.planetOutMsgs[0].recorder(mc.sec2nano(dt_sampling))
        self.asteroidTruthRec = dyn_model.ephemConverter.ephemOutMsgs[0].recorder(mc.sec2nano(dt_sampling))
        self.scTruthRec = dyn_model.scObject.scStateOutMsg.recorder(mc.sec2nano(dt_sampling))

        # Add logging tasks
        self.AddModelToTask(dyn_model.taskName, self.asteroidEphemRec)
        self.AddModelToTask(dyn_model.taskName, self.asteroidTruthRec)
        self.AddModelToTask(dyn_model.taskName, self.scTruthRec)

    # This subclass retrieves basilisk on-orbit data
    def orbit_data(self, parameters, outputs):
        # Set shape object
        masconfit_bsk = masconFit.MasconFit()
        shape = masconfit_bsk.shape
        shape.initPolyhedron(parameters.asteroid.xyz_vert.tolist(),
                             parameters.asteroid.order_face.tolist())

        # Retrieve simulation times
        t = np.array(self.scTruthRec.times() * mc.NANO2SEC)
        n = len(t)

        # Retrieve spacecraft truth position and velocity
        pos_BN_N0 = np.array(self.scTruthRec.r_BN_N)
        vel_BN_N0 = np.array(self.scTruthRec.v_BN_N)

        # Retrieve asteroid truth position, velocity, mrp and angular velocity
        pos_PN_N0 = np.array(self.asteroidTruthRec.r_BdyZero_N)
        vel_PN_N0 = np.array(self.asteroidTruthRec.v_BdyZero_N)
        mrp_PN0 = np.array(self.asteroidTruthRec.sigma_BN)
        angvel_PN0_P = np.array(self.asteroidTruthRec.omega_BN_B)

        # Compute spacecraft truth state relative to asteroid in inertial frame
        pos_BP_N0 = np.array(pos_BN_N0 - pos_PN_N0)
        vel_BP_N0 = np.array(vel_BN_N0 - vel_PN_N0)

        # Preallocate spacecraft truth position and velocity in asteroid inertial and rotating
        pos_BP_N1 = np.zeros((n, 3))
        vel_BP_N1 = np.zeros((n, 3))
        pos_BP_P = np.zeros((n, 3))
        vel_BP_P = np.zeros((n, 3))

        # Preallocate gravity acceleration in asteroid and inertial frames
        acc_BP_P = np.zeros((n, 3))

        # Preallocate inhomogeneous gravity acceleration in asteroid frame
        accHigh_BP_P = np.zeros((n, 3))

        # Preallocate asteroid position, orientation and Sun's direction
        pos_PS_N1 = np.zeros((n, 3))
        eul323_PN0 = np.zeros((n, 3))
        e_SP_P = np.zeros((n, 3))

        # Preallocate spacecraft truth radius and altitude
        r_BP = np.zeros(n)
        h_BP = np.zeros(n)

        # Direction cosine matrix from ecliptic and equatorial inertial frames
        dcm_N1N0 = parameters.asteroid.dcm_N1N0

        # Spacecraft orientation in asteroid frame
        mrp_BP = np.zeros((n, 3))

        # Loop through simulation times
        for i in range(n):
            # Retrieve dcm between inertial and asteroid frame
            dcm_PN0 = rbk.MRP2C(mrp_PN0[i, 0:3])
            dcm_PN1 = np.matmul(dcm_PN0, dcm_N1N0.T)

            # Rotate spacecraft truth position and velocity to asteroid frame
            pos_BP_P[i, 0:3] = dcm_PN0.dot(pos_BP_N0[i, 0:3])
            vel_BP_P[i, 0:3] = dcm_PN0.dot(vel_BP_N0[i, 0:3]) - np.cross(angvel_PN0_P[i, 0:3],
                                                                         pos_BP_P[i, 0:3])

            # Fill spacecraft truth radius and altitude
            r_BP[i] = np.linalg.norm(pos_BP_P[i, 0:3])
            h_BP[i] = shape.computeAltitude(pos_BP_P[i, 0:3].tolist())

            # Fill asteroid orientation
            eul323_PN0[i, 0:3] = rbk.MRP2Euler323(mrp_PN0[i, 0:3])

            # Fill gravity acceleration
            acc_BP_P[i, 0:3] = \
                np.array(self.DynModels.asteroid.poly.computeField(pos_BP_P[i, 0:3])).reshape(3)

            # Rotate truth position and velocity to equatorial inertial frame
            pos_BP_N1[i, 0:3] = np.dot(dcm_N1N0, pos_BP_N0[i, 0:3])
            vel_BP_N1[i, 0:3] = np.dot(dcm_N1N0, vel_BP_N0[i, 0:3])

            ## Rotate gravity acceleration to equatorial inertial frame
            #acc_CA_N1[i, 0:3] = np.dot(dcm_N1N0, acc_CA_N0[i, 0:3])

            # Rotate asteroid position to equatorial inertial frame
            pos_PS_N1[i, 0:3] = np.dot(dcm_N1N0, pos_PN_N0[i, 0:3])

            # Rotate Sun's direction to equatorial inertial frame
            e_SP_P[i, 0:3] = np.dot(dcm_PN0, -pos_PN_N0[i, 0:3] / np.linalg.norm(pos_PN_N0[i, 0:3]))

            # Generate "fake" attitude to point camera towards asteroid
            lon = np.arctan2(pos_BP_P[i, 1], pos_BP_P[i, 0])
            lat = np.arcsin(pos_BP_P[i, 2] / r_BP[i])
            mrp_BP[i, 0:3] = rbk.euler3232MRP([lon, -(np.pi/2 + lat), np.pi/2])

        # Save outputs simulation variables related to spacecraft
        outputs.groundtruth.t = t
        outputs.groundtruth.pos_BP_N0 = pos_BP_N0
        outputs.groundtruth.vel_BP_N0 = vel_BP_N0
        outputs.groundtruth.pos_BP_N1 = pos_BP_N1
        outputs.groundtruth.vel_BP_N1 = vel_BP_N1
        outputs.groundtruth.pos_BP_P = pos_BP_P
        outputs.groundtruth.vel_BP_P = vel_BP_P
        outputs.groundtruth.mrp_BP = mrp_BP
        #outputs.groundtruth.acc_CA_N0 = acc_CA_N0
        outputs.groundtruth.acc_BP_P = acc_BP_P
        outputs.groundtruth.accHigh_BP_P = accHigh_BP_P
        outputs.groundtruth.r_BP = r_BP
        outputs.groundtruth.h_BP = h_BP

        # Save outputs simulation variables related to asteroid
        outputs.groundtruth.pos_PS_N1 = pos_PS_N1
        outputs.groundtruth.e_SP_P = e_SP_P
        outputs.groundtruth.mrp_PN0 = mrp_PN0

    # This subclass creates ejecta data
    def ejecta_data(self, parameters, outputs):
        # Set shape object
        masconfit_bsk = masconFit.MasconFit()
        shape = masconfit_bsk.shape
        shape.initPolyhedron(parameters.asteroid.xyz_vert.tolist(),
                             parameters.asteroid.order_face.tolist())

        # Retrieve number of orbits, ejecta and its max radius
        n_orbits = parameters.n_segments
        n_ejecta = parameters.asteroid.n_ejecta
        rmax_ejecta = parameters.asteroid.rmax_ejecta

        # Retrieve time and asteroid rotating frame positions
        t = outputs.groundtruth.t
        pos_BP_P = outputs.groundtruth.pos_BP_P

        # Preallocate ejecta positions and gravity accelerations
        pos_EP_P = np.zeros((n_orbits, n_ejecta, 3))
        acc_EP_P = np.zeros((n_orbits, n_ejecta, 3))

        # Preallocate ejecta radius and altitude
        r_EP = np.zeros((n_orbits, n_ejecta))
        h_EP = np.zeros((n_orbits, n_ejecta))

        # Initialize random seed and loop through orbits
        np.random.seed(0)
        for i in range(n_orbits):
            # Obtain random indexes for each orbit
            idx = np.where(np.logical_and(t >= parameters.times_dmcukf[i, 0],
                                          t <= parameters.times_dmcukf[i, 1]))[0]
            idx_ejecta = np.random.choice(idx, size=n_ejecta, replace=False)

            # Loop through ejecta
            for j in range(n_ejecta):
                # Initialize flag and counter
                flag = False
                cont = 0

                # Loop until an ejecta outside asteroid is found
                while not flag:
                    # Try ejecta radius
                    if cont < parameters.asteroid.maxiter_ejecta:
                        r = np.random.uniform(0, rmax_ejecta)
                    else:
                        r = parameters.asteroid.rmaxiter_ejecta

                    # Try ejecta position and compute normalized Laplacian
                    pos = r * (pos_BP_P[idx_ejecta[j], 0:3]
                               / np.linalg.norm(pos_BP_P[idx_ejecta[j], 0:3]))
                    lap = shape.computeLaplacian(pos.tolist())

                    # Save ejecta sample if outside of asteroid
                    if abs(lap) < 2*np.pi:
                        # Save ejecta position and gravity acceleration
                        pos_EP_P[i, j, 0:3] = pos
                        acc_EP_P[i, j, 0:3] = \
                            np.array(self.DynModels.asteroid.poly.computeField(pos)).reshape(3)

                        # Save ejecta radius and altitude
                        r_EP[i, j] = r
                        h_EP[i, j] = shape.computeAltitude(pos.tolist())

                        # Set flag to break loop
                        flag = True

                    # Add one to iteration counter
                    cont += 1

        # Save outputs simulation variables related to ejecta
        outputs.groundtruth.pos_EP_P = pos_EP_P
        outputs.groundtruth.acc_EP_P = acc_EP_P
        outputs.groundtruth.r_EP = r_EP
        outputs.groundtruth.h_EP = h_EP

    # This subclass creates dense data
    def dense_data(self, parameters, outputs):
        # Set shape object
        masconfit_bsk = masconFit.MasconFit()
        shape = masconfit_bsk.shape
        shape.initPolyhedron(parameters.asteroid.xyz_vert.tolist(),
                             parameters.asteroid.order_face.tolist())

        # Set times
        t = np.linspace(parameters.times_groundtruth[0], parameters.times_groundtruth[-1],
                        int((parameters.times_groundtruth[-1]-parameters.times_groundtruth[0]) / parameters.dmcukf_rate))
        pos_BP_P = np.zeros((len(t), 3))
        acc_BP_P = np.zeros((len(t), 3))
        rmax = 30 * 1e3

        # Create position-gravity acceleration time
        for i in range(len(t)):
            # Set exterior point flag to false
            flag_ext = False
            while not flag_ext:
                # Do a random position sample
                lat = np.random.uniform(-np.pi/2, np.pi/2)
                lon = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, rmax)
                pos = r * np.array([np.cos(lat) * np.cos(lon),
                                    np.cos(lat) * np.sin(lon),
                                    np.sin(lat)])

                # Save only if it is an interior point
                lap = shape.computeLaplacian(pos.tolist())
                if abs(lap) < 2*np.pi:
                    pos_BP_P[i, 0:3] = pos
                    acc_BP_P[i, 0:3] = np.array(self.DynModels.asteroid.poly.computeField(pos.tolist())).reshape(3)
                    flag_ext = True

        # Save generated dataset
        outputs.groundtruth.t = t
        outputs.groundtruth.pos_BP_P = pos_BP_P
        outputs.groundtruth.acc_BP_P = acc_BP_P


def run_groundtruth(scenario, parameters, outputs):
    # Set flight mode
    scenario.modeRequest = 'standby'

    # Initialize simulation
    scenario.InitializeSimulation()

    # Set type of ground truth data creation
    if parameters.grav_est.data_type == 'orbit':
        # Configure run time and execute simulation
        tf = mc.sec2nano(parameters.times_groundtruth[-1])
        scenario.ConfigureStopTime(tf)
        scenario.ExecuteSimulation()

        # Save generated data
        scenario.orbit_data(parameters, outputs)
        scenario.ejecta_data(parameters, outputs)
    elif parameters.grav_est.data_type == 'dense':
        # Create dense data
        scenario.dense_data(parameters, outputs)
