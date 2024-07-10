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
        self.t0_list = []
        self.y0_list = []

        # Initialize list of truth
        self.t_list = []
        self.pos_list = []
        self.vel_list = []

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

        # List of solutions
        if not self.sol_list:
            self.sol_list = [[] for _ in range(n)]

        # Loop through maneuvers
        for i in range(n):
            # Initialize solver
            self.init_solver((0, self.tf[i]),
                             self.pos0[i, 0:3],
                             self.posf[i, 0:3])

            # Warm start if x0, y0 not available
            if not self.sol_list[i]:
                # Solve Kepler transfer
                sol_kep = self.solver.solve_bvp_kepler()

                # Store only if there is not collision
                if sol_kep.has_collide:
                    sol_kep = None
                self.sol_list[i] = sol_kep

            # Solve with full asteroid
            if self.sol_list[i] is not None:
                sol = self.solver.solve_bvp(self.sol_list[i].x,
                                            self.sol_list[i].y)

                # Store in list
                self.sol_list[i] = sol

    # This method initializes propagator instance
    def init_propagator(self):
        self.propagator = Propagator(self.asteroid_truth,
                                     n_eval=100)

    # This method propagates transfers
    def propagate_transfers(self):
        # Get number of transfers
        n = len(self.pos0)

        # Initialize propagator
        self.init_propagator()

        # Create times, positions and
        # velocities lists
        self.t_list = []
        self.pos_list = []
        self.vel_list = []

        # Loop through maneuvers
        for i in range(n):
            if self.sol_list[i] is not None:
                vel0 = self.sol_list[i].vel0
                t, pos, vel = self.propagator.propagate([0, self.tf[i]],
                                                        self.pos0[i, 0:3],
                                                        vel0)
            else:
                t, pos, vel = None, None, None

            # Save into list
            self.t_list.append(t)
            self.pos_list.append(pos)
            self.vel_list.append(vel)
