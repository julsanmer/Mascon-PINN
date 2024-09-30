import torch

from src.gravRegression.nn.bcModels.model_bc import ModelBC


# This is the mascon gravity as a boundary model
class Mascon(ModelBC):
    def __init__(self, mu_M=None, xyz_M=None, device='cpu'):
        # Set mascon variables
        self.n_M = len(mu_M)
        self.mu_M = torch.from_numpy(mu_M).to(device, dtype=torch.float32)
        self.xyz_M = torch.tensor(xyz_M).to(device, dtype=torch.float32)

        # Set device
        self.device = 'cpu'

    # This method computes potential
    def compute_acc(self, pos):
        # Preallocate potential
        acc = torch.zeros_like(pos)

        # Loop through masses
        for k in range(self.n_M):
            # Compute distance vector to the k-th mass
            dpos_k = pos - self.xyz_M[k, 0:3].t()  # Shape: [n, 3]

            # Compute the distance (norm of the distance vector)
            dr_k = torch.norm(dpos_k, dim=1)

            # Add to gravity
            acc += -self.mu_M[k] * dpos_k / dr_k[:, None]**3

        return acc

    # This method computes potential
    def compute_U(self, pos):
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
