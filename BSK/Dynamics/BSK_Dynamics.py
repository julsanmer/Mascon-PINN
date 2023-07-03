from Basilisk.architecture import messaging
from Basilisk.simulation import (ephemerisConverter, planetEphemeris, radiationPressure, spacecraft)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import simIncludeGravBody

from Basilisk import __path__
bskPath = __path__[0]


class BSKDynamicModels:
    """
    General bskSim simulation class that sets up the spacecraft simulation configuration.
    """
    def __init__(self, SimBase, parameters):
        # Define process name, task name and task time-step
        self.processName = SimBase.DynamicsProcessName
        self.taskName = "DynamicsTask"
        self.processTasksTimeStep = mc.sec2nano(parameters.dyn_rate)

        # Add task to dynamics process
        SimBase.dynProc.addTask(SimBase.CreateNewTask(self.taskName, self.processTasksTimeStep))

        # Copy modules in the dynamics class
        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.gravBodyEphem = planetEphemeris.PlanetEphemeris()
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        if parameters.flag_sun:
            self.radiationPressure = radiationPressure.RadiationPressure()
        self.scObject = spacecraft.Spacecraft()

        # Preallocate asteroid
        self.asteroid = None

        # Initialize all dynamics modules
        self.initialize_dynamics(parameters)

        # Assign initialized modules to tasks
        SimBase.AddModelToTask(self.taskName, self.gravBodyEphem, ModelPriority=100)
        SimBase.AddModelToTask(self.taskName, self.ephemConverter, ModelPriority=99)
        SimBase.AddModelToTask(self.taskName, self.scObject, ModelPriority=98)
        if parameters.flag_sun:
            SimBase.AddModelToTask(self.taskName, self.radiationPressure, ModelPriority=97)

    # ------------------------------------------------------------------------------------------- #
    # These are module-initialization methods
    def set_spacecraft(self, parameters_spacecraft):
        """
        Specify the spacecraft parameters
        """
        # Set name and spacecraft mass
        self.scObject.ModelTag = "bskSat"
        self.scObject.hub.mHub = parameters_spacecraft.mass

        # Attach gravity bodies and srp
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(self.gravFactory.gravBodies.values()))
        self.scObject.addDynamicEffector(self.radiationPressure)

    def set_ephemeris(self):
        """
        Specify ephemeris of gravitational bodies
        """
        # Set name
        self.gravBodyEphem.ModelTag = 'asteroidEphemeris'
        self.gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(['asteroid']))

    def set_gravity(self, parameters_asteroid, flag_sun):
        """
        Specify what gravitational bodies to include in the simulation
        """
        # Set asteroid name and its gravity parameter
        name_asteroid = parameters_asteroid.name
        mu_asteroid = parameters_asteroid.mu

        # Set central body according to solar perturbations
        if not flag_sun:
            self.asteroid.isCentralBody = True

        # Add polyhedron gravity
        if parameters_asteroid.grav == 'poly':
            self.asteroid = self.gravFactory.createCustomGravObject(name_asteroid, mu_asteroid)
            file_poly = parameters_asteroid.polyFile
            self.asteroid.usePolyhedral = True
            simIncludeGravBody.loadPolyFromFile(file_poly, self.asteroid.poly)

        # Subscribe asteroid gravity to its ephemeris
        self.asteroid.planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])

    def set_solarperturbations(self, parameters_spacecraft):
        """
        Specify solar perturbations
        """
        # Create Sun gravity
        self.gravFactory.createSun()

        # Create a sun spice message, zero it out, required by srp
        sun_message = messaging.SpicePlanetStateMsgPayload()
        self.sunPlanetStateMsg = messaging.SpicePlanetStateMsg()
        self.sunPlanetStateMsg.write(sun_message)

        # Set spacecraft srp properties and connect Sun message
        self.radiationPressure.area = parameters_spacecraft.srpArea
        self.radiationPressure.coefficientReflection = parameters_spacecraft.CR
        self.radiationPressure.sunEphmInMsg.subscribeTo(self.sunPlanetStateMsg)

    def set_ephemerisconverter(self):
        """
        Specify ephemeris converter
        """
        # Create an ephemeris converter
        self.ephemConverter.ModelTag = 'ephemConverter'
        self.ephemConverter.addSpiceInputMsg(self.gravBodyEphem.planetOutMsgs[0])

    def initialize_dynamics(self, parameters):
        """
        Initialize all the dynamics modules.
        """
        # Fill variables for each dynamics module (order matters)
        self.set_ephemeris()
        self.set_gravity(parameters.asteroid, parameters.flag_sun)
        if parameters.flag_sun:
            self.set_solarperturbations(parameters.spacecraft)
        self.set_ephemerisconverter()
        self.set_spacecraft(parameters.spacecraft)
