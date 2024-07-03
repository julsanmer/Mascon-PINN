import numpy as np
import os
import inspect

from Basilisk.simulation import planetEphemeris
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport

# Import master classes: simulation base class and scenario base class
from src.bskObjects.bskSim import BSKSimulation

# Get current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


# Create your own scenario child class
class Propagator(BSKSimulation):
    def __init__(self, asteroid, sc, collision=False):
        super().__init__()

        # Set rates
        self.dt_sample = sc.data.dt_sample

        # Set dynamics subclass
        self.set_DynModel(asteroid, sc)

        # Preallocate loggers
        self.asteroidEphemRec, self.asteroidRec = [], []
        self.scRec = []

        # Set dynamics initial condition and loggers
        self.set_dyn_initial_condition(asteroid, sc)
        self.set_loggers(dt_sampling=self.dt_sample)

        # Set collision event
        if collision:
            self.set_collision_event(asteroid)

    # This method sets initial conditions for asteroid and spacecraft
    def set_dyn_initial_condition(self, asteroid, sc, flag_sun=False):
        # Get dynamics model
        dyn_model = self.get_DynModel()

        if flag_sun:
            # Set asteroid orbital elements
            oe_asteroid = planetEphemeris.ClassicElementsMsgPayload()
            oe_asteroid.a = asteroid.oe[0]
            oe_asteroid.e = asteroid.oe[1]
            oe_asteroid.i = asteroid.oe[2]
            oe_asteroid.Omega = asteroid.oe[3]
            oe_asteroid.omega = asteroid.oe[4]
            oe_asteroid.f = asteroid.oe[5]
            dyn_model.gravBodyEphem.planetElements = \
                planetEphemeris.classicElementVector([oe_asteroid])

        # Set asteroid north pole and rotation
        ra = asteroid.eul313[0]
        dec = asteroid.eul313[1]
        lst0 = asteroid.eul313[2]
        rot_period = asteroid.rot_period
        dyn_model.gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([ra])
        dyn_model.gravBodyEphem.declination = planetEphemeris.DoubleVector([dec])
        dyn_model.gravBodyEphem.lst0 = planetEphemeris.DoubleVector([lst0])
        dyn_model.gravBodyEphem.rotRate = \
            planetEphemeris.DoubleVector([360 * mc.D2R / rot_period])

        # Set required asteroid parameters to initialize spacecraft
        mu_asteroid = asteroid.mu
        dcm_PN = rbk.euler3232C([ra, np.pi/2-dec, lst0])
        omega_PN_P = np.array([0, 0, 360*mc.D2R / rot_period])
        if flag_sun:
            r_PN_N, v_PN_N = \
                orbitalMotion.elem2rv(orbitalMotion.MU_SUN * (1000.**3), oe_asteroid)
        else:
             r_PN_N, v_PN_N = np.zeros(3), np.zeros(3)

        # Set spacecraft orbit elements
        oe_spacecraft = orbitalMotion.ClassicElements()
        oe_spacecraft.a = sc.oe[0]
        oe_spacecraft.e = sc.oe[1]
        oe_spacecraft.i = sc.oe[2]
        oe_spacecraft.Omega = sc.oe[3]
        oe_spacecraft.omega = sc.oe[4]
        oe_spacecraft.f = sc.oe[5]
        r_BP, v_BP = orbitalMotion.elem2rv(mu_asteroid,
                                           oe_spacecraft)

        # Rotate spacecraft initial condition
        # to inertial coordinates
        r_BP_N = dcm_PN.transpose().dot(r_BP)
        v_BP_N = dcm_PN.transpose().dot(v_BP)

        # Assume asteroid-centred depending on Sun's flag
        if flag_sun:
            r_BN_N = r_BP_N + r_PN_N
            v_BN_N = v_BP_N + v_PN_N
        else:
            r_BN_N = r_BP_N
            v_BN_N = v_BP_N

        # Set spacecraft initial condition in BSK
        dyn_model.sc.hub.r_CN_NInit = r_BN_N
        dyn_model.sc.hub.v_CN_NInit = v_BN_N

    # This method adds a collision event
    def set_collision_event(self, asteroid):
        # Shape object
        self.shape = asteroid.shape

        # Define objects for collision evaluation
        self.sc_state = self.DynModels.sc.scStateOutMsg
        self.asteroid_state = self.DynModels.ephemConverter.ephemOutMsgs[0]
        self.rbk = rbk
        self.ut = unitTestSupport

        # Collision conditional in string
        f1 = f"self.shape.check_exterior("
        f2 = "self.rbk.MRP2C(self.asteroid_state.read().sigma_BN).dot("
        #f3 = "np.array(self.sc_state.read().r_BN_N)-np.array(self.asteroid_state.read().r_BdyZero_N)))"
        f3 = "self.ut.EigenVector3d2np(self.posRef.getState())-np.array(self.asteroid_state.read().r_BdyZero_N)))"
        f4 = "== False"
        f_collision = f1 + f2 + f3 + f4

        # Event to terminate the simulation
        self.createNewEvent("Collision", mc.sec2nano(self.dt_sample), True,
                            [f_collision], [], terminal=True)

    # This method adds logging tasks to basilisk
    def set_loggers(self, dt_sampling=1.):
        # Get dynamics model
        dyn_model = self.get_DynModel()

        # Set asteroid and spacecraft recorders
        self.asteroidEphemRec = \
            dyn_model.gravBodyEphem.planetOutMsgs[0].recorder(mc.sec2nano(dt_sampling))
        self.asteroidRec = \
            dyn_model.ephemConverter.ephemOutMsgs[0].recorder(mc.sec2nano(dt_sampling))
        self.scRec = \
            dyn_model.sc.scStateOutMsg.recorder(mc.sec2nano(dt_sampling))

        # Add logging tasks
        self.AddModelToTask(dyn_model.taskName, self.asteroidEphemRec)
        self.AddModelToTask(dyn_model.taskName, self.asteroidRec)
        self.AddModelToTask(dyn_model.taskName, self.scRec)

    # This method saves ouputs
    def save_outputs(self, asteroid, sc):
        # Retrieve simulation times
        t = np.array(self.scRec.times() * mc.NANO2SEC)
        n = len(t)

        # Retrieve spacecraft position and velocity
        # in centred inertial frame N0
        pos_BN_N0 = np.array(self.scRec.r_BN_N)
        vel_BN_N0 = np.array(self.scRec.v_BN_N)

        # Retrieve asteroid position, velocity, mrp and angular velocity
        # in centred inertial frame N0
        pos_PN_N0 = np.array(self.asteroidRec.r_BdyZero_N)
        vel_PN_N0 = np.array(self.asteroidRec.v_BdyZero_N)
        mrp_PN0 = np.array(self.asteroidRec.sigma_BN)
        angvel_PN0_P = np.array(self.asteroidRec.omega_BN_B)

        # Compute spacecraft position and velocity w.r.t.
        # asteroid in centred inertial frame N0
        pos_BP_N0 = np.array(pos_BN_N0 - pos_PN_N0)
        vel_BP_N0 = np.array(vel_BN_N0 - vel_PN_N0)

        # Preallocate spacecraft position and velocity
        # w.r.t. asteroid in inertial N1 and rotating P
        pos_BP_N1, vel_BP_N1 = np.zeros((n, 3)), np.zeros((n, 3))
        pos_BP_P, vel_BP_P = np.zeros((n, 3)), np.zeros((n, 3))

        # Preallocate radius, altitude and orientation
        # w.r.t. asteroid
        r_BP, h_BP = np.zeros(n), np.zeros(n)
        mrp_BP = np.zeros((n, 3))

        # Preallocate gravity and potential
        acc_BP_P = np.zeros((n, 3))
        U = np.zeros(n)

        # Preallocate asteroid states
        eul323_PN0 = np.zeros((n, 3))
        pos_PS_N1 = np.zeros((n, 3))
        e_SP_P = np.zeros((n, 3))

        # Retrieve dcm of ecliptic w.r.t. equatorial
        # inertial frame
        dcm_N1N0 = asteroid.dcm_N1N0

        # Loop through times
        for i in range(n):
            # Retrieve dcm between inertial and asteroid frame
            dcm_PN0 = rbk.MRP2C(mrp_PN0[i, 0:3])
            dcm_PN1 = np.matmul(dcm_PN0, dcm_N1N0.T)

            # Transform spacecraft position and velocity to equatorial
            # inertial frame N1
            pos_BP_N1[i, 0:3] = np.dot(dcm_N1N0, pos_BP_N0[i, 0:3])
            vel_BP_N1[i, 0:3] = np.dot(dcm_N1N0, vel_BP_N0[i, 0:3])

            # Transform spacecraft position and velocity to
            # rotating asteroid frame P
            pos_BP_P[i, 0:3] = np.dot(dcm_PN0, pos_BP_N0[i, 0:3])
            vel_BP_P[i, 0:3] = np.dot(dcm_PN0, vel_BP_N0[i, 0:3]) \
                               - np.cross(angvel_PN0_P[i, 0:3], pos_BP_P[i, 0:3])

            # Fill spacecraft radius and altitude
            r_BP[i] = np.linalg.norm(pos_BP_P[i, 0:3])
            h_BP[i] = asteroid.shape.compute_altitude(pos_BP_P[i, 0:3])

            # Fill gravity acceleration and potential
            acc_BP_P[i, 0:3] = np.zeros(3)
            U[i] = 0.
            for gravity in self.DynModels.asteroid:
                acc_BP_P[i, 0:3] += np.array(gravity.gravityModel.computeField(
                    pos_BP_P[i, 0:3])).reshape(3)
                U[i] += gravity.gravityModel.computePotentialEnergy(
                    pos_BP_P[i, 0:3].tolist())

            # Fill asteroid orientation
            eul323_PN0[i, 0:3] = rbk.MRP2Euler323(mrp_PN0[i, 0:3])

            # Rotate asteroid position to equatorial inertial frame
            pos_PS_N1[i, 0:3] = np.dot(dcm_N1N0, pos_PN_N0[i, 0:3])

            # Rotate Sun's direction to equatorial inertial frame
            e_SP_P[i, 0:3] = np.dot(dcm_PN0, -pos_PN_N0[i, 0:3]
                                    / np.linalg.norm(pos_PN_N0[i, 0:3]))

            # Generate "fake" attitude to point camera towards asteroid
            lon = np.arctan2(pos_BP_P[i, 1], pos_BP_P[i, 0])
            lat = np.arcsin(pos_BP_P[i, 2] / r_BP[i])
            mrp_BP[i, 0:3] = rbk.euler3232MRP([lon,
                                               -(np.pi/2 + lat),
                                               np.pi/2])

        # Save spacecraft data
        sc_data = sc.data
        sc_data.t = t
        sc_data.pos_BP_N0, sc_data.vel_BP_N0 = pos_BP_N0, vel_BP_N0
        sc_data.pos_BP_N1, sc_data.vel_BP_N1 = pos_BP_N1, vel_BP_N1
        sc_data.pos_BP_P, sc_data.vel_BP_P = pos_BP_P, vel_BP_P
        sc_data.mrp_BP = mrp_BP
        sc_data.r_BP, sc_data.h_BP = r_BP, h_BP
        sc_data.acc_BP_P = acc_BP_P
        sc_data.U = U

        # Save asteroid data
        asteroid_data = asteroid.data
        asteroid_data.t = t
        asteroid_data.pos_PS_N0 = pos_PN_N0
        asteroid_data.vel_PS_N0 = vel_PN_N0
        asteroid_data.pos_PS_N1 = pos_PS_N1
        asteroid_data.angvel_PN0_P = angvel_PN0_P
        asteroid_data.eul323_PN0 = eul323_PN0
        asteroid_data.e_SP_P = e_SP_P
        asteroid_data.mrp_PN0 = mrp_PN0
