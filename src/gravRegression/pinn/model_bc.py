import numpy as np
import torch


# This is the mascon gravity as a boundary model
class Mascon_BC:
    def __init__(self, mu_M=None, xyz_M=None, device='cpu'):
        # Set mascon variables
        self.n_M = len(mu_M)
        self.mu_M = torch.from_numpy(mu_M).to(device, dtype=torch.float32)
        self.xyz_M = torch.tensor(xyz_M).to(device, dtype=torch.float32)

        # Set device
        self.device = 'cpu'

    # This method computes potential
    def compute_potential(self, pos):
        # If it is not a tensor
        if not torch.is_tensor(pos):
            pos = torch.from_numpy(pos)

        # Preallocate potential
        U = torch.zeros(pos.shape[0],
                        device=self.device,
                        dtype=torch.float32)

        # Loop through masses
        for k in range(self.n_M):
            # Compute distance with kth mass
            dpos_k = pos - self.xyz_M[k, 0:3].t()
            dr_k = torch.norm(dpos_k, dim=1)

            # Add to potential
            U += self.mu_M[k] / dr_k

        return U

    # This method computes potential
    def compute_acc(self, pos):
        # If it is not a tensor
        if not torch.is_tensor(pos):
            pos = torch.from_numpy(pos)

        # Preallocate potential
        acc = torch.zeros_like(pos)

        # Loop through masses
        for k in range(self.n_M):
            # Compute distance vector to the k-th mass
            dpos_k = pos - self.xyz_M[k, 0:3].t()  # Shape: [n, 3]

            # Compute the distance (norm of the distance vector)
            dr_k = torch.norm(dpos_k, dim=1)  # Shape: [n]

            # Add to potential
            acc += -self.mu_M[k] * dpos_k / dr_k[:, None] ** 3  # dr_k[:, None] makes it [n, 1]

        return acc


# This is the spherical harmonics gravity as a boundary model
class Spherharm_BC:
    def __init__(self, mu=None, rE=None, deg=None,
                 C=None, S=None, device='cpu'):
        # Set standard gravity, normalization radius
        # and degree
        self.mu = mu
        self.rE = rE
        self.deg = deg

        # Set spherical harmonics
        self.C = C
        self.S = S

    # This method computes potential
    def compute_potential(self, pos):
        # Compute Pines coordinates
        s = pos[:, 0] / r
        t = pos[:, 1] / r
        u = pos[:, 2] / r

        # Set R, I and rho
        R0, R1, R2 = 1, s, s**2 - t**2
        I1, I2 = t, 2*s*t
        rho1 = self.rE / r
        rho2 = rho1 * (self.rE / r)

        # Set A
        A00, A10, A11 = 1, u*np.sqrt(3), np.sqrt(3)
        A20, A21, A22 = \
            np.sqrt(5/4) * (3*u**2 - 1), u*np.sqrt(15), np.sqrt(15/4)

        # Compute potential
        U00 = A00
        U10 = rho1 * A10 * self.C[1, 0] * R0
        U11 = rho1 * A11 * (self.C[1, 1] * R1
                            + self.S[1, 1] * I1)
        U20 = rho2 * A20 * self.C[2, 0] * R0
        U21 = rho2 * A21 * (self.C[2, 1] * R1
                            + self.S[2, 1] * I1)
        U22 = rho2 * A22 * (self.C[2, 2] * R2
                            + self.S[2, 2] * I2)

        # Compute total potential
        U = (U00 + U10 + U11
             + U20 + U21 + U22) * (self.mu / r)

        return U
