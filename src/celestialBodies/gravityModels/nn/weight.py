import torch


# This the class for weights function
class Weight:
    def __init__(self, k, r_switch, R):
        # Switch parameters
        self.R = R
        self.k = k
        self.r_switch = r_switch

    # This method computes weights
    def compute_w(self, r):
        # Compute switch function
        h = self.k * (r - self.r_switch) / self.R
        H = (1 + torch.tanh(h)) / 2

        # Compute both weights
        w1 = H
        w2 = 1 - H

        return w1, w2
