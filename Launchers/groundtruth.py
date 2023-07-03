import numpy as np
import os
import inspect

from Basilisk.utilities import orbitalMotion, RigidBodyKinematics
from Basilisk.utilities import macros as mc
from Basilisk.simulation import planetEphemeris
from Basilisk.ExternalModules import gravEst

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
        dcm_AN = RigidBodyKinematics.euler3232C([ra, np.pi/2 - dec, lst0])
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

        # Set spacecraft initial condition in Basilisk
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
        # Set gravity module
        gravest_bsk = gravEst.GravEst()
        gravest_bsk.poly.nFacet = parameters.asteroid.n_face
        gravest_bsk.poly.nVertex = parameters.asteroid.n_vert
        gravest_bsk.poly.orderFacet = (parameters.asteroid.order_face - 1).tolist()
        gravest_bsk.poly.xyzVertex = parameters.asteroid.xyz_vert.tolist()
        gravest_bsk.poly.initializeParameters()

        # Retrieve simulation times
        t = np.array(self.scTruthRec.times() * mc.NANO2SEC)
        n = len(t)

        # Retrieve spacecraft truth position and velocity
        pos_CN_N = np.array(self.scTruthRec.r_BN_N)
        vel_CN_N = np.array(self.scTruthRec.v_BN_N)

        # Retrieve asteroid truth position, velocity, mrp and angular velocity
        pos_AN_N = np.array(self.asteroidTruthRec.r_BdyZero_N)
        vel_AN_N = np.array(self.asteroidTruthRec.v_BdyZero_N)
        mrp_AN = np.array(self.asteroidTruthRec.sigma_BN)
        angvel_AN_A = np.array(self.asteroidTruthRec.omega_BN_B)

        # Compute spacecraft truth state relative to asteroid in inertial frame
        pos_CA_N = np.array(pos_CN_N - pos_AN_N)
        vel_CA_N = np.array(vel_CN_N - vel_AN_N)

        # Preallocate spacecraft truth position and velocity
        pos_CA_A = np.zeros((n, 3))
        vel_CA_A = np.zeros((n, 3))

        # Preallocate gravity acceleration in asteroid and inertial frames
        acc_CA_A = np.zeros((n, 3))
        acc_CA_N = np.zeros((n, 3))

        # Preallocate inhomogeneous gravity acceleration in asteroid frame
        accHigh_CA_A = np.zeros((n, 3))

        # Preallocate asteroid position, orientation and Sun's direction
        pos_AS_N = np.zeros((n, 3))
        eul323_AN = np.zeros((n, 3))
        e_SA_A = np.zeros((n, 3))

        # Preallocate spacecraft truth radius and altitude
        r_CA = np.zeros(n)
        h_CA = np.zeros(n)

        # Direction cosine matrix from ecliptic and equatorial inertial frames
        dcm_N1N0 = RigidBodyKinematics.euler3232C([
            parameters.asteroid.ra, np.pi/2 - parameters.asteroid.dec, 0])

        # Loop through simulation times
        for i in range(n):
            # Retrieve dcm between inertial and asteroid frame
            dcm_AN = RigidBodyKinematics.MRP2C(mrp_AN[i, 0:3])

            # Rotate spacecraft truth position and velocity to asteroid frame
            pos_CA_A[i, 0:3] = dcm_AN.dot(pos_CA_N[i, 0:3])
            vel_CA_A[i, 0:3] = dcm_AN.dot(vel_CA_N[i, 0:3]) - np.cross(angvel_AN_A[i, 0:3],
                                                                       pos_CA_A[i, 0:3])

            # Fill spacecraft truth radius and altitude
            r_CA[i] = np.linalg.norm(pos_CA_A[i, 0:3])
            h_CA[i] = gravest_bsk.poly.computeAltitude(pos_CA_A[i, 0:3].tolist())

            # Fill asteroid orientation
            eul323_AN[i, 0:3] = RigidBodyKinematics.MRP2Euler323(mrp_AN[i, 0:3])

            # Fill gravity acceleration
            acc_CA_A[i, 0:3] = np.array(self.DynModels.asteroid.poly.computeField(pos_CA_A[i, 0:3])).reshape(3) \
                               + self.DynModels.asteroid.mu*pos_CA_A[i,0:3]/np.linalg.norm(pos_CA_A[i, 0:3])**3

            # Rotate truth position and velocity to equatorial inertial frame
            pos_CA_N[i, 0:3] = np.dot(dcm_N1N0, pos_CA_N[i, 0:3])
            vel_CA_N[i, 0:3] = np.dot(dcm_N1N0, vel_CA_N[i, 0:3])

            # Rotate gravity acceleration to equatorial inertial frame
            acc_CA_N[i, 0:3] = np.dot(dcm_N1N0, acc_CA_N[i, 0:3])

            # Rotate asteroid position to equatorial inertial frame
            pos_AS_N[i, 0:3] = np.dot(dcm_N1N0, pos_AN_N[i, 0:3])

            # Rotate Sun's direction to equatorial inertial frame
            e_SA_A[i, 0:3] = np.dot(dcm_AN, -pos_AN_N[i, 0:3] / np.linalg.norm(pos_AN_N[i, 0:3]))

        # Save outputs simulation variables related to spacecraft
        outputs.groundtruth.t = t
        outputs.groundtruth.pos_CA_N = pos_CA_N
        outputs.groundtruth.vel_CA_N = vel_CA_N
        outputs.groundtruth.pos_CA_A = pos_CA_A
        outputs.groundtruth.vel_CA_A = vel_CA_A
        outputs.groundtruth.acc_CA_N = acc_CA_N
        outputs.groundtruth.acc_CA_A = acc_CA_A
        outputs.groundtruth.accHigh_CA_A = accHigh_CA_A
        outputs.groundtruth.r_CA = r_CA
        outputs.groundtruth.h_CA = h_CA

        # Save outputs simulation variables related to asteroid
        outputs.groundtruth.pos_AS_N = pos_AS_N
        outputs.groundtruth.e_SA_A = e_SA_A
        outputs.groundtruth.eul323_AN = eul323_AN

    # This subclass creates ejecta data
    def ejecta_data(self, parameters, outputs):
        # Set gravity module
        gravest_bsk = gravEst.GravEst()
        gravest_bsk.poly.nFacet = parameters.asteroid.n_face
        gravest_bsk.poly.nVertex = parameters.asteroid.n_vert
        gravest_bsk.poly.orderFacet = (parameters.asteroid.order_face - 1).tolist()
        gravest_bsk.poly.xyzVertex = parameters.asteroid.xyz_vert.tolist()
        gravest_bsk.poly.initializeParameters()

        # Retrieve number of orbits, ejecta and its max radius
        n_orbits = parameters.n_segments
        n_ejecta = parameters.asteroid.n_ejecta
        rmax_ejecta = parameters.asteroid.rmax_ejecta

        # Retrieve time and asteroid rotating frame positions
        t = outputs.groundtruth.t
        pos_CA_A = outputs.groundtruth.pos_CA_A

        # Preallocate ejecta positions and gravity accelerations
        pos_EA_A = np.zeros((n_orbits, n_ejecta, 3))
        acc_EA_A = np.zeros((n_orbits, n_ejecta, 3))

        # Preallocate ejecta radius and altitude
        r_EA = np.zeros((n_orbits, n_ejecta))
        h_EA = np.zeros((n_orbits, n_ejecta))

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
                        r_try = np.random.uniform(0, rmax_ejecta)
                    else:
                        r_try = parameters.asteroid.rmaxiter_ejecta

                    # Try ejecta position and compute normalized Laplacian
                    pos_try = r_try*(pos_CA_A[idx_ejecta[j], 0:3] / np.linalg.norm(pos_CA_A[idx_ejecta[j], 0:3]))
                    lap_try = gravest_bsk.poly.computeLaplacian([pos_try.tolist(), [0, 0, 0]])

                    # Save ejecta sample if outside of asteroid
                    if abs(lap_try[0][0]) < 2*np.pi:
                        # Save ejecta position and gravity acceleration
                        pos_EA_A[i, j, 0:3] = pos_try
                        acc_EA_A[i, j, 0:3] = np.array(self.DynModels.asteroid.poly.computeField(pos_try)
                                                       ).reshape(3)

                        # Save ejecta radius and altitude
                        r_EA[i, j] = r_try
                        h_EA[i, j] = gravest_bsk.poly.computeAltitude(pos_try.tolist())

                        # Set flag to break loop
                        flag = True

                    # Add one to iteration counter
                    cont += 1

        # Save outputs simulation variables related to ejecta
        outputs.groundtruth.pos_EA_A = pos_EA_A
        outputs.groundtruth.acc_EA_A = acc_EA_A
        outputs.groundtruth.r_EA = r_EA
        outputs.groundtruth.h_EA = h_EA

    # This subclass creates dense data
    def dense_data(self, parameters, outputs):
        # Create ancillary gravity model
        gravest_bsk = gravEst.GravEst()
        gravest_bsk.poly.nVertex = parameters.asteroid.n_vert
        gravest_bsk.poly.nFacet = parameters.asteroid.n_face
        gravest_bsk.poly.xyzVertex = parameters.asteroid.xyz_vert
        gravest_bsk.poly.orderFacet = parameters.asteroid.order_face
        gravest_bsk.poly.initializeParameters()

        # Set times
        t = np.linspace(parameters.times_groundtruth[0], parameters.times_groundtruth[-1],
                        int((parameters.times_groundtruth[-1]-parameters.times_groundtruth[0]) / parameters.dmcukf_rate))
        r_CA_A = np.zeros((len(t), 3))
        a_A = np.zeros((len(t), 3))
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
                r_i = r * np.array([np.cos(lat) * np.cos(lon),
                                    np.cos(lat) * np.sin(lon),
                                    np.sin(lat)])

                # Save only if it is an interior point
                lap = gravest_bsk.poly.computeLaplacian([r_i.tolist(), [0, 0, 0]])
                if abs(lap[0][0]) < 2*np.pi:
                    r_CA_A[i, 0:3] = r_i
                    a_A[i, 0:3] = np.array(self.DynModels.asteroid.poly.computeField(r_i.tolist())).reshape(3) \
                                  + parameters.asteroid.mu * r_i / np.linalg.norm(r_i)**3
                    flag_ext = True
                print(i)

        # Save generated dataset
        outputs.groundtruth.t = t
        outputs.groundtruth.r_CA_A = r_CA_A
        outputs.groundtruth.a_A = a_A

        #h = np.zeros(len(pos))
        #for i in range(len(pos)):
        #    h[i] = gravest_bsk.poly.computeAltitude(pos[i, 0:3].tolist())
        #data.h = h


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
