from Basilisk.simulation import planetNav, simpleNav
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport as sp
from Basilisk.ExternalModules import cameraNav4

class BSKFswModels:
    """Defines the bskSim FSW class"""
    def __init__(self, SimBase, params):
        # Define process name and default time-step for all FSW tasks defined later on
        self.processName = SimBase.FSWProcessName
        self.processTasksTimeStep = mc.sec2nano(params.fswRate)

        # Create module data
        self.cameraNavMeas = cameraNav4.CameraNav4()
        self.planetNavMeas = planetNav.PlanetNav()
        self.simpleNavMeas = simpleNav.SimpleNav()

        # Initialize all modules
        self.InitAllFSWObjects(SimBase, params)

        # Create task
        SimBase.fswProc.addTask(SimBase.CreateNewTask("measTask", self.processTasksTimeStep))

        # Assign initialized modules to tasks
        SimBase.AddModelToTask("measTask", self.cameraNavMeas, ModelPriority=97)
        SimBase.AddModelToTask("measTask", self.simpleNavMeas, ModelPriority=96)
        SimBase.AddModelToTask("measTask", self.planetNavMeas, ModelPriority=95)

        # Create events to be called for triggering GN&C maneuvers
        SimBase.fswProc.disableAllTasks()

        SimBase.createNewEvent("initiateStandby", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'standby'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('measTask')",
                                "self.setAllButCurrentEventActivity('initiateStandby', True)"])


    # ------------------------------------------------------------------------------------------- #
    # These are module-initialization methods
    def SetCameraNav(self, SimBase, sensorsParams):
        """Set the navigation sensor object."""
        self.cameraNavMeas.ModelTag = 'CameraNav'

        #self.cameraNavMeas.f = sensorsParams.f
        #self.cameraNavMeas.camFOV = sensorsParams.camFOV
        #self.cameraNavMeas.nPixel = sensorsParams.nPixel

        self.cameraNavMeas.nxPixel = sensorsParams.nxPixel
        self.cameraNavMeas.nyPixel = sensorsParams.nyPixel
        self.cameraNavMeas.wPixel = sensorsParams.wPixel
        self.cameraNavMeas.f = sensorsParams.f

        self.cameraNavMeas.maskangleCam = sensorsParams.maskangleCam
        self.cameraNavMeas.maskangleSun = sensorsParams.maskangleSun

        self.cameraNavMeas.dxyzLandmark = sensorsParams.dxyzLandmark

        self.cameraNavMeas.body.xyzVertex = sensorsParams.xyzVertex
        self.cameraNavMeas.body.orderFacet = sensorsParams.orderFacet
        self.cameraNavMeas.body.idxLandmark = sensorsParams.idxLandmark

        self.cameraNavMeas.ephemerisInMsg.subscribeTo(SimBase.DynModels.ephemConverter.ephemOutMsgs[0])
        self.cameraNavMeas.scStateInMsg.subscribeTo(SimBase.DynModels.scObject.scStateOutMsg)


    def SetPlanetNav(self, SimBase, sensorsParams):
        """Set the navigation sensor object."""
        self.planetNavMeas.ModelTag = 'PlanetNav'

        self.planetNavMeas.PMatrix = sensorsParams.Psmallbody
        self.planetNavMeas.walkBounds = sp.np2EigenVectorXd(sensorsParams.walkSmallbody)

        self.planetNavMeas.ephemerisInMsg.subscribeTo(SimBase.DynModels.ephemConverter.ephemOutMsgs[0])


    def SetSimpleNav(self, SimBase, sensorsParams):
        """Set the navigation sensor object."""
        self.simpleNavMeas.ModelTag = 'SimpleNav'

        self.simpleNavMeas.PMatrix = sensorsParams.Pspacecraft
        self.simpleNavMeas.walkBounds = sp.np2EigenVectorXd(sensorsParams.walkSpacecraft)

        self.simpleNavMeas.scStateInMsg.subscribeTo(SimBase.DynModels.scObject.scStateOutMsg)


    # Global call to initialize every module
    def InitAllFSWObjects(self, SimBase, params):
        """Initialize all the FSW objects"""
        self.SetPlanetNav(SimBase, params.sensors)
        self.SetCameraNav(SimBase, params.sensors)
        self.SetSimpleNav(SimBase, params.sensors)
