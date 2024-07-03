import numpy as np
import os
import inspect

from Basilisk.simulation import planetEphemeris
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros as mc

from Basilisk.fswAlgorithms import masconFit

# Import master classes: simulation base class and scenario base class
from src.classes.inner.bskObjects.bskSim import BSKScenario

from src.groundtruth.position_sample import alt_sample,\
    ell_sample, rad_sample

# Get current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


# Create your own scenario child class
class Propagator(BSKScenario):
    def __init__(self, asteroid, sc):
        super().__init__()

        # Set dynamics subclass
        self.set_DynModel(asteroid, sc)

        # Preallocate loggers
        self.asteroidEphemRec, self.asteroidRec = [], []
        self.scRec = []

        # Set dynamics initial condition and loggers
        self.configure_dyn_initial_conditions(asteroid, sc)
        self.set_loggers(dt_sampling=sc.dt_sampling)

    # This method sets initial conditions for asteroid and spacecraft
    def set_dyn_initial_condition(self, asteroid, sc, flag_sun=True):
        # Get dynamics model
        dyn_model = self.get_DynModel()

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
        dcm_AN = rbk.euler3232C([ra, np.pi/2-dec, lst0])
        omega_AN_A = np.array([0, 0, 360*mc.D2R / rot_period])
        r_AN_N, v_AN_N = \
            orbitalMotion.elem2rv(orbitalMotion.MU_SUN * (1000.**3), oe_asteroid)

        # Set spacecraft orbit elements
        oe_spacecraft = orbitalMotion.ClassicElements()
        oe_spacecraft.a = sc.oe_[0]
        oe_spacecraft.e = sc.oe[1]
        oe_spacecraft.i = sc.oe[2]
        oe_spacecraft.Omega = sc.oe[3]
        oe_spacecraft.omega = sc.oe[4]
        oe_spacecraft.f = sc.oe[5]
        r_CA, v_CA = orbitalMotion.elem2rv(mu_asteroid, oe_spacecraft)

        # Rotate spacecraft initial condition
        # to inertial coordinates
        r_CA_N = dcm_AN.transpose().dot(r_CA)
        v_CA_N = dcm_AN.transpose().dot(v_CA)

        # Assume asteroid-centred depending on Sun's flag
        if flag_sun:
            r_CN_N = r_CA_N + r_AN_N
            v_CN_N = v_CA_N + v_AN_N
        else:
            r_CN_N = r_CA_N
            v_CN_N = v_CA_N

        # Set spacecraft initial condition in BSK
        dyn_model.scObject.hub.r_CN_NInit = r_CN_N
        dyn_model.scObject.hub.v_CN_NInit = v_CN_N

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
            dyn_model.scObject.scStateOutMsg.recorder(mc.sec2nano(dt_sampling))

        # Add logging tasks
        self.AddModelToTask(dyn_model.taskName, self.asteroidEphemRec)
        self.AddModelToTask(dyn_model.taskName, self.asteroidRec)
        self.AddModelToTask(dyn_model.taskName, self.scRec)

    # This method saves outputs
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

        # Retrieve dcm of ecliptic w.r.t. equatorial
        # inertial frame
        dcm_N1N0 = asteroid.dcm_N1N0

        # Loop through simulation times
        for i in range(n):
            # Retrieve dcm between inertial and asteroid frame
            dcm_PN0 = rbk.MRP2C(mrp_PN0[i, 0:3])
            dcm_PN1 = np.matmul(dcm_PN0, dcm_N1N0.T)

            # Transform spacecraft position and velocity to
            # rotating asteroid frame P
            pos_BP_P[i, 0:3] = dcm_PN0.dot(pos_BP_N0[i, 0:3])
            vel_BP_P[i, 0:3] = dcm_PN0.dot(vel_BP_N0[i, 0:3]) \
                               - angvel_PN0_P[i, 0:3].cross(pos_BP_P[i, 0:3])

            # Fill spacecraft radius and altitude
            r_BP[i] = np.linalg.norm(pos_BP_P[i, 0:3])
            h_BP[i] = asteroid.shape.compute_altitude(pos_BP_P[i, 0:3])

            # Fill gravity acceleration and potential
            acc_BP[i, 0:3] = np.zeros(3)
            U = 0.
            for gravity in self.DynModels.asteroid_bsk:
                acc_BP_P[i, 0:3] += np.array(gravity.computeField(pos_BP_P[i, 0:3])).reshape(3)
                U[i] += gravity.computePotentialEnergy(pos_BP_P[i, 0:3].tolist())

            # Fill asteroid orientation
            eul323_PN0[i, 0:3] = rbk.MRP2Euler323(mrp_PN0[i, 0:3])

            # Rotate position and velocity to equatorial inertial frame
            pos_BP_N1[i, 0:3] = np.dot(dcm_N1N0, pos_BP_N0[i, 0:3])
            vel_BP_N1[i, 0:3] = np.dot(dcm_N1N0, vel_BP_N0[i, 0:3])

            # Rotate asteroid position to equatorial inertial frame
            pos_PS_N1[i, 0:3] = np.dot(dcm_N1N0, pos_PN_N0[i, 0:3])

            # Rotate Sun's direction to equatorial inertial frame
            e_SP_P[i, 0:3] = np.dot(dcm_PN0, -pos_PN_N0[i, 0:3]
                                    / np.linalg.norm(pos_PN_N0[i, 0:3]))

            # Generate "fake" attitude to point camera towards asteroid
            lon = np.arctan2(pos_BP_P[i, 1], pos_BP_P[i, 0])
            lat = np.arcsin(pos_BP_P[i, 2] / r_BP[i])
            mrp_BP[i, 0:3] = rbk.euler3232MRP([lon, -(np.pi/2 + lat), np.pi/2])

    # # This subclass creates ejecta data
    # def ejecta_data(self, parameters, outputs):
    #     # Get asteroid properties
    #     asteroid = parameters.asteroid
    #
    #     # Set shape object
    #     masconfit_bsk = masconFit.MasconFit()
    #     shape = masconfit_bsk.shapeModel
    #     shape.xyzVertex = asteroid.xyz_vert.tolist()
    #     shape.orderFacet = asteroid.order_face.tolist()
    #     shape.initializeParameters()
    #
    #     # Retrieve ejecta and its max radius
    #     n_ejecta = asteroid.n_ejecta
    #     rmax_ejecta = asteroid.rmax_ejecta
    #
    #     # Preallocate ejecta positions and gravity accelerations
    #     pos_EP_P = np.zeros((n_ejecta, 3))
    #     acc_EP_P = np.zeros((n_ejecta, 3))
    #
    #     # Preallocate ejecta radius and altitude
    #     r_EP = np.zeros(n_ejecta)
    #     h_EP = np.zeros(n_ejecta)
    #
    #     # Initialize random seed and loop through orbits
    #     np.random.seed(0)
    #     for i in range(n_ejecta):
    #         # Initialize flag and counter
    #         flag_out = False
    #         cont = 0
    #
    #         # Loop until an ejecta outside asteroid is found
    #         while not flag_out:
    #             # Try ejecta radius
    #             if asteroid.dist_ejecta == 'rad':
    #                 pos = rad_sample(rmax_ejecta)
    #             elif asteroid.dist_ejecta == 'ell':
    #                 pos = ell_sample(rmax_ejecta, asteroid.axes)
    #             elif asteroid.dist_ejecta == 'alt':
    #                 pos = alt_sample(rmax_ejecta, asteroid.xyz_face)
    #
    #             # Try ejecta position and compute normalized Laplacian
    #             isExterior = shape.isExterior(pos.tolist())
    #
    #             # Save ejecta sample if outside of asteroid
    #             if isExterior:
    #                 # Save ejecta position and gravity acceleration
    #                 pos_EP_P[i, 0:3] = pos
    #                 acc_EP_P[i, 0:3] = np.array(
    #                     self.DynModels.asteroid[0].poly.computeField(pos)).reshape(3)
    #                 if parameters.asteroid.add_mascon:
    #                     acc_EP_P[i, 0:3] += np.array(
    #                         self.DynModels.asteroid[1].mascon.computeField(pos)).reshape(3)
    #
    #                 # Save ejecta radius and altitude
    #                 r_EP[i] = np.linalg.norm(pos)
    #                 h_EP[i] = shape.computeAltitude(pos.tolist())
    #
    #                 # Set flag to break loop
    #                 flag_out = True
    #
    #             # Add one to iteration counter
    #             cont += 1
    #
    #     # Save outputs simulation variables related to ejecta
    #     states_truth = outputs.groundtruth.states
    #     states_truth.pos_EP_P = pos_EP_P
    #     states_truth.acc_EP_P = acc_EP_P
    #     states_truth.r_EP = r_EP
    #     states_truth.h_EP = h_EP
