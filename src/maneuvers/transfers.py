import numpy as np

from src.maneuvers.solver import Solver
from src.maneuvers.propagator import Propagator

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This class stores transfers results
class Transfers:
    def __init__(self, asteroid, pos0, posf, tf):
        # Set a model asteroid and a truth one
        self.asteroid = asteroid
        self.asteroid_truth = None

        # Initialize solver and propagator
        self.solver = None
        self.propagator = None

        # Set bc conditions
        self.pos0 = pos0
        self.posf = posf
        self.tf = tf

        # Create an array to store solver cpu time
        self.t_cpu = np.zeros(len(pos0))

        # Initialize list of solutions
        self.sol_list = []

    # Initialize solver
    def init_solver(self, t_span, pos0, posf):
        # Creates an instance of bvp solver
        self.solver = Solver(self.asteroid,
                             t_span,
                             pos0,
                             posf)

    # Solve transfers
    def solve_transfers(self):
        # Get number of transfers
        n = len(self.pos0)

        # Initialize solver list
        sol_list = []

        # Loop through
        for i in range(n):
            # Initialize solver
            self.init_solver((0, self.tf[i]),
                             self.pos0[i, 0:3],
                             self.posf[i, 0:3])

            # Solve warm start
            sol_kep = self.solver.solve_bvp_kepler()

            # Solve with full asteroid
            if not sol_kep.has_collide:
                sol = self.solver.solve_bvp(sol_kep.x, sol_kep.y)
            else:
                sol = None

            # Append to list
            self.sol_list.append(sol)

    # This method initializes propagator instance
    def init_propagator(self):
        self.propagator = Propagator(self.asteroid_truth,
                                     n_eval=100)

    # This method propagates transfers
    def propagate_transfers(self):
        # Get number of transfers
        n = len(self.pos0)

        self.init_propagator()

        self.t_list = []
        self.pos_list = []
        self.vel_list = []

        for i in range(n):
            if self.sol_list[i] is not None:
                vel0 = self.sol_list[i].y[3:6, 0]
                t, pos, vel = self.propagator.propagate([0, self.tf[i]],
                                                        self.pos0[i, 0:3],
                                                        vel0)
            else:
                t, pos, vel = None, None, None

        a = 4






