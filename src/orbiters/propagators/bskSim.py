import numpy as np

from Basilisk.architecture import messaging
from Basilisk.simulation import gravityEffector
from Basilisk.simulation import (ephemerisConverter, planetEphemeris,
                                 radiationPressure, spacecraft, svIntegrators)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import simIncludeGravBody

# Get current file path
import sys, os, inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

# Import Dynamics and FSW models
sys.path.append(path + '/models')


# This is the main bskSim simulation class
class BSKSimulation(SimulationBaseClass.SimBaseClass):
    """Main bskSim simulation class"""
    def __init__(self):
        # Create a sim module as an empty container
        SimulationBaseClass.SimBaseClass.__init__(self)

        # Set progress bar
        self.SetProgressBar(True)

        # Preallocate dynamics models and process
        self.DynModels = []
        self.DynamicsProcessName = None
        self.dynProc = None
        self.dynamics_added = False

    # This method returns attached dynamics
    def get_DynModel(self):
        assert (self.dynamics_added is True), "It is mandatory to use a dynamics model as an argument"
        return self.DynModels

    # This method sets dynamics model
    def set_DynModel(self, asteroid, sc):
        # Set dynamics process and create models
        self.dynamics_added = True
        self.DynamicsProcessName = 'DynamicsProcess'  # Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName)  # Create process
        self.DynModels = self.BSKDynamicModels(self, asteroid, sc)  # Create Dynamics class

    # This method initializes and resets BSK modules
    def init_sim(self):
        self.InitializeSimulation()

        # Get access to dynManager position and velocity
        self.posRef = self.DynModels.sc.dynManager.getStateObject("hubPosition")
        self.velRef = self.DynModels.sc.dynManager.getStateObject("hubVelocity")

    # This method simulates during period of time
    def simulate(self, tf):
        # Configure stop time and propagate
        self.ConfigureStopTime(mc.sec2nano(tf))
        self.ExecuteSimulation()

    # This method sets initial condition
    def set_dyn_initial_condition(self, asteroid, sc, flag_sun):
        pass

    # This method adds loggers
    def set_loggers(self, dt_sample):
        pass

    # This method saves outputs
    def save_outputs(self, asteroid, sc):
        pass

    # Sets the dynamics modules
    class BSKDynamicModels:
        def __init__(self, SimBase, asteroid, sc, flag_sun=True):
            # Define process name, task name and task time-step
            self.processName = SimBase.DynamicsProcessName
            self.taskName = "DynamicsTask"
            self.processTasksTimeStep = mc.sec2nano(sc.dt_dyn)

            # Add task to dynamics process
            SimBase.dynProc.addTask(SimBase.CreateNewTask(self.taskName,
                                                          self.processTasksTimeStep))

            # Copy modules in the dynamics class
            self.ephemConverter = ephemerisConverter.EphemerisConverter()
            self.gravBodyEphem = planetEphemeris.PlanetEphemeris()
            self.gravFactory = simIncludeGravBody.gravBodyFactory()
            flag_sun = False
            if flag_sun:
                self.radiationPressure = radiationPressure.RadiationPressure()
            self.sc = spacecraft.Spacecraft()

            # Set integrator
            self.integrator = svIntegrators.svIntegratorRKF45(self.sc)
            self.integrator.relTol = 1e-8
            self.integrator.absTol = 1e-8
            self.sc.setIntegrator(self.integrator)

            # Preallocate asteroid object
            self.grav_body = []

            # Initialize all dynamics modules
            self.initialize_dynamics(asteroid, sc, flag_sun)

            # Assign initialized modules to tasks
            SimBase.AddModelToTask(self.taskName, self.gravBodyEphem, ModelPriority=100)
            SimBase.AddModelToTask(self.taskName, self.ephemConverter, ModelPriority=99)
            SimBase.AddModelToTask(self.taskName, self.sc, ModelPriority=98)
            if flag_sun:
                SimBase.AddModelToTask(self.taskName, self.radiationPressure, ModelPriority=97)

        # This method initializes all dynamics
        def initialize_dynamics(self, grav_body, sc, flag_sun):
            # Fill variables for each dynamics module (order matters)
            self.set_ephemeris()
            self.set_gravity(grav_body)
            if flag_sun:
                self.set_solarperturbations(sc)
            self.set_ephemerisconverter()
            self.set_spacecraft(sc, flag_sun)

        # This method initializes asteroid ephemeris
        def set_ephemeris(self):
            # Set name
            self.gravBodyEphem.ModelTag = 'asteroidEphemeris'
            self.gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(['asteroid']))

        # This method initializes ephemeris converter
        def set_ephemerisconverter(self):
            # Create an ephemeris converter
            self.ephemConverter.ModelTag = 'ephemConverter'
            self.ephemConverter.addSpiceInputMsg(self.gravBodyEphem.planetOutMsgs[0])

        # This method initializes asteroid gravity
        def set_gravity(self, grav_body):
            # Preallocate gravity model indexes
            # and total mu
            idx = 0
            mu = 0

            # Loop through gravity models
            for gravity in grav_body.gravity:
                # Mascon gravity
                if gravity.name == 'mascon':
                    mu += np.sum(gravity.mu_M)
                    mascon = self.gravFactory.createCustomGravObject('mascon', np.sum(gravity.mu_M))
                    self.grav_body.append(mascon)
                    self.grav_body[idx].useMasconGravityModel()
                    self.grav_body[idx].mascon.muMascon = gravity.mu_M.tolist()
                    self.grav_body[idx].mascon.xyzMascon = gravity.xyz_M.tolist()
                    self.grav_body[idx].planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
                    idx += 1
                # Spherical harmonics gravity
                elif gravity.name == 'spherharm':
                    self.grav_body.append(
                        self.gravFactory.createCustomGravObject('spherharm', mu=gravity.mu))
                    self.grav_body[idx].useSphericalHarmParams = True
                    self.grav_body[idx].spherHarm.cBar = gravityEffector.MultiArray(gravity.C)
                    self.grav_body[idx].spherHarm.sBar = gravityEffector.MultiArray(gravity.S)
                    self.grav_body[idx].mu = gravity.mu
                    self.grav_body[idx].spherHarm.maxDeg = gravity.deg
                    self.grav_body[idx].radEquator = gravity.rE
                    self.grav_body[idx].planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
                    idx += 1
                # PINN gravity
                elif gravity.name == 'pinn':
                    self.grav_body.append(
                        self.gravFactory.createCustomGravObject('pinn', 0.))
                    self.grav_body[idx].usePINN2GravityModel()
                    self.grav_body[idx].pinn.PINNPath = gravity.file_torch
                    self.grav_body[idx].planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
                    idx += 1
                # Polyhedron gravity
                elif gravity.name == 'poly':
                    mu += gravity.mu
                    self.grav_body.append(
                        self.gravFactory.createCustomGravObject('poly', gravity.mu))
                    self.grav_body[idx].usePolyhedralGravityModel(gravity.file)
                    self.grav_body[idx].planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
                    idx += 1
                else:
                    raise Exception('Unrecognized gravity in BSK')

        # This method initializes spacecraft
        def set_spacecraft(self, sc, flag_sun):
            # Set name and spacecraft mass
            self.sc.ModelTag = "bskSat"
            self.sc.hub.mHub = sc.mass

            # Attach gravity bodies and srp
            self.sc.gravField.gravBodies = \
                spacecraft.GravBodyVector(list(self.gravFactory.gravBodies.values()))
            if flag_sun:
                self.sc.addDynamicEffector(self.radiationPressure)

        # This method initializes solar perturbations
        def set_solarperturbations(self, sc):
            # Create Sun gravity
            self.gravFactory.createSun()

            # Create a sun spice message, zero it out, required by srp
            sun_message = messaging.SpicePlanetStateMsgPayload()
            self.sunPlanetStateMsg = messaging.SpicePlanetStateMsg()
            self.sunPlanetStateMsg.write(sun_message)

            # Set spacecraft srp properties and connect Sun message
            self.radiationPressure.area = sc.srp_area
            self.radiationPressure.coefficientReflection = sc.CR
            self.radiationPressure.sunEphmInMsg.subscribeTo(self.sunPlanetStateMsg)
