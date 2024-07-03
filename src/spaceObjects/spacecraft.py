# Spacecraft method
class Spacecraft:
    def __init__(self, oe=None):
        # Set mass and srp properties
        self.mass = 750.0
        self.srp_area = 1.1
        self.CR = 1.2

        # Set initial orbital elements
        self.oe = oe

        # Set dyn rate
        self.dt_dyn = 1

        # Set data
        self.data = self.Data()

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
