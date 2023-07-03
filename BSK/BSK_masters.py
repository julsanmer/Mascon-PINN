# Import architectural modules
from Basilisk.utilities import SimulationBaseClass

# Get current file path
import sys, os, inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

# Import Dynamics and FSW models
sys.path.append(path + '/models')


class BSKSim(SimulationBaseClass.SimBaseClass):
    """Main bskSim simulation class"""
    def __init__(self):
        # Create a sim module as an empty container
        SimulationBaseClass.SimBaseClass.__init__(self)

        self.SetProgressBar(True)
        self.DynModels = []
        self.FSWModels = []
        self.DynamicsProcessName = None
        self.FSWProcessName = None
        self.dynProc = None
        self.fswProc = None

        self.dynamics_added = False
        self.fsw_added = False

    def get_DynModel(self):
        assert (self.dynamics_added is True), "It is mandatory to use a dynamics model as an argument"
        return self.DynModels

    def set_DynModel(self, dynModel, scenarioParams):
        self.dynamics_added = True
        self.DynamicsProcessName = 'DynamicsProcess'  # Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName)  # Create process
        self.DynModels = dynModel.BSKDynamicModels(self, scenarioParams)  # Create Dynamics and FSW classes

    def get_FswModel(self):
        assert (self.fsw_added is True), "A flight software model has not been added yet"
        return self.FSWModels

    def set_FswModel(self, fswModel, scenarioParams):
        self.fsw_added = True
        self.FSWProcessName = "FSWProcess"  # Create simulation process name
        self.fswProc = self.CreateNewProcess(self.FSWProcessName)  # Create process
        self.FSWModels = fswModel.BSKFswModels(self, scenarioParams)  # Create Dynamics and FSW classes


class BSKScenario(object):
    def __init__(self):
        self.name = "scenario"

    def configure_initial_conditions(self):
        """
            Developer must override this method in their BSK_Scenario derived subclass.
        """
        pass

    def log_outputs(self):
        """
            Developer must override this method in their BSK_Scenario derived subclass.
        """
        pass

    def pull_outputs(self):
        """
            Developer must override this method in their BSK_Scenario derived subclass.
        """
        pass
