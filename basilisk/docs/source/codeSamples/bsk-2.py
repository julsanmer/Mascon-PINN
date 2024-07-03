#
#  ISC License
#
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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

from Basilisk.moduleTemplates import cModuleTemplate
from Basilisk.moduleTemplates import cppModuleTemplate
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros


def run():
    """
    Illustration of adding Basilisk modules to a task
    """

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")

    # create the dynamics task and specify the integration update time
    dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", macros.sec2nano(5.)))

    # create copies of the Basilisk modules
    mod1 = cModuleTemplate.cModuleTemplate()
    mod1.ModelTag = "cModule1"

    mod2 = cppModuleTemplate.CppModuleTemplate()
    mod2.ModelTag = "cppModule2"

    mod3 = cModuleTemplate.cModuleTemplate()
    mod3.ModelTag = "cModule3"

    scSim.AddModelToTask("dynamicsTask", mod1)
    scSim.AddModelToTask("dynamicsTask", mod2, 10)
    scSim.AddModelToTask("dynamicsTask", mod3, 5)

    #  initialize Simulation:
    scSim.InitializeSimulation()
    print("InitializeSimulation() completed...")

    #   configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(macros.sec2nano(5.0))
    scSim.ExecuteSimulation()

    return


if __name__ == "__main__":
    run()
