from timeit import default_timer as timer
from src.orbiters.propagators.propagator import Propagator


# Spacecraft method
class Spacecraft:
    def __init__(self, grav_body=None):
        # Set mass and srp properties
        self.mass = 750.0
        self.srp_area = 1.1
        self.CR = 1.2

        # Set dyn rate
        self.dt_dyn = 1

        # Set data
        self.data = self.Data()

        # Define propagation variables
        self.grav_body = grav_body
        self.propagator = None

        # Define propagation cpu time
        self.t_cpu = []

    # Propagate different orbits
    def propagate(self, oe0, tf):
        print('------- Initiating propagation -------')

        # Create propagator instance and initialize
        propagator = Propagator(self.grav_body,
                                self,
                                oe0)
        propagator.init_sim()

        # Start measuring cpu time
        t_start = timer()

        # Propagate
        propagator.simulate(tf)

        # End measuring cpu time
        t_end = timer()
        self.t_cpu = t_end - t_start

        # Save data
        propagator.save_outputs(self.grav_body,
                                self)

        # Print finish
        print('------- Finished propagation -------')

        ## Delete swigpy objects
        #self.grav_body.delete_swigpy()

    # This stores sim data
    class Data:
        def __init__(self):
            # Preallocate sampling and time
            self.dt_sample = []
            self.t = []

            # Preallocate position and velocity
            self.pos_BP_N0, self.vel_BP_N0 = [], []
            self.pos_BP_N1, self.vel_BP_N1 = [], []
            self.pos_BP_P, self.vel_BP_P = [], []

            # Preallocate orientation in asteroid frame
            self.mrp_BP = []

            # Preallocate radius and altitude
            self.r_BP, self.h_BP = [], []

            # Preallocate gravity acceleration
            self.acc_BP_N0 = []
            self.acc_BP_P = []
            self.accHigh_BP_P = []

            # Preallocate potential
            self.U = []

            # Preallocate propagation error
            self.pos_err = []
