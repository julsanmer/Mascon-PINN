from src.gravRegression.nn.bcModels.model_bc import ModelBC


# This is the spherical harmonics gravity as a boundary model
class Spherharm(ModelBC):
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
    def compute_U(self, pos):
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
