# 
#  ISC License
# 
#  Copyright (c) 2024, Autonomous Vehicle Systems Lab, University of Colorado Boulder
# 
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
# 
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# 
# 

import pytest

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
from Basilisk.moduleTemplates import autoCModule

@pytest.mark.parametrize("accuracy", [1e-12])
@pytest.mark.parametrize("param1, param2", [
     (1, 1)
    ,(1, 3)
])

def test_autoCModule(show_plots, param1, param2, accuracy):
    r"""
    **Validation Test Description**

    Compose a general description of what is being tested in this unit test script.

    **Test Parameters**

    Discuss the test parameters used.

    Args:
        param1 (int): Dummy test parameter for this parameterized unit test
        param2 (int): Dummy test parameter for this parameterized unit test
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    Here discuss what variables and states are being checked. 
    """
    [testResults, testMessage] = autoCModuleTestFunction(show_plots, param1, param2, accuracy)
    assert testResults < 1, testMessage


def autoCModuleTestFunction(show_plots, param1, param2, accuracy):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(0.5)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = autoCModule.autoCModule()
    module.ModelTag = "autoCModuleTag"
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Configure blank module input messages
    someInMsgData = messaging.AttRefMsgPayload()
    someInMsg = messaging.AttRefMsg().write(someInMsgData)

    some2InMsgData = messaging.AttRefMsgPayload()
    some2InMsg = messaging.AttRefMsg().write(some2InMsgData)

    anotherInMsgData = messaging.CSSConfigMsgPayload()
    anotherInMsg = messaging.CSSConfigMsg().write(anotherInMsgData)

    # subscribe input messages to module
    module.someInMsg.subscribeTo(someInMsg)
    module.some2InMsg.subscribeTo(some2InMsg)
    module.anotherInMsg.subscribeTo(anotherInMsg)

    # setup output message recorder objects
    some2OutMsgRec = module.some2OutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, some2OutMsgRec)
    someOutMsgRec = module.someOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, someOutMsgRec)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))
    unitTestSim.ExecuteSimulation()

    # pull module data and make sure it is correct

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_autoCModule(False, 1, 1, 1e-12)


