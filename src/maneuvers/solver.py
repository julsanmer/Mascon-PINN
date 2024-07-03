import numpy as np

from scipy.integrate import solve_bvp


# This class solves transfer from A to B
class Solver:
    def __init__(self, asteroid, t_span, pos0, posf):
        # Boundary conditions
        self.t0, self.tf = t_span[0], t_span[1]
        self.pos0, self.posf = pos0, posf

        # Compute adimensional variables
        self.t_ad = t_span[1] - t_span[0]
        self.r_ad = np.linalg.norm(posf - pos0)

        # Set asteroid instance
        self.asteroid = asteroid

        # Set evaluation model
        self.model = 'full'

    # This method encodes boundary condition
    def _bc(self, y0, yf):
        dy = np.concatenate((y0[0:3] - self.pos0/self.r_ad,
                             yf[0:3] - self.posf/self.r_ad))

        return dy

    # This method encodes dynamics
    def _dynamics(self, t, y):
        # Dimensionalize position and velocity
        pos = y[0:3] * self.r_ad
        vel = y[3:6] * self.r_ad/self.t_ad

        # Preallocate gravity for mesh
        _, n_nodes = y.shape

        # Preallocate acceleration
        acc = np.zeros((3, n_nodes))

        # Fill acceleration
        for i in range(n_nodes):
            if self.model == 'full':
                acc[0:3, i] = self.asteroid.compute_gravity(pos[0:3, i])
            elif self.model == 'kepler':
                acc[0:3, i] = -self.asteroid.mu * pos[0:3, i] / np.linalg.norm(pos[0:3, i]) ** 3

        # Angular velocity
        angvel = np.array([0,
                           0,
                           2*np.pi / self.asteroid.rot_period])
        angvel_mesh = angvel[:, np.newaxis].T

        # Dynamics derivatives
        pos_dot = vel
        vel_dot = - 2 * np.cross(angvel_mesh, vel.T).T \
                  - np.cross(angvel_mesh, np.cross(angvel_mesh, pos.T)).T \
                  + acc

        # Dimensionalize derivatives
        pos_dot *= self.t_ad / self.r_ad
        vel_dot *= self.t_ad**2 / self.r_ad

        # Concatenate output
        y_dot = np.concatenate((pos_dot,
                                vel_dot))

        return y_dot

    # Check if there are collisions
    def check_collision(self, pos):
        # Initialize collision flag
        has_collide = False

        # Loop through trajectory
        for i in range(len(pos)):
            # Compute exterior flag
            is_exterior = self.asteroid.shape.check_exterior(pos[i, 0:3])

            # Return if there is a collision
            if not is_exterior:
                has_collide = True

                return has_collide

        return has_collide

    # This method solves transfer bvp with Kepler
    def solve_bvp_kepler(self):
        # Define initial mesh
        t0 = np.linspace(0, 1, 100)
        y0 = np.ones((6, t0.shape[0]))

        # Set Kepler model flag
        self.model = 'kepler'

        # Solve bvp transfer
        sol = solve_bvp(self._dynamics,
                        self._bc,
                        t0,
                        y0,
                        verbose=0,
                        max_nodes=400,
                        tol=1e-4)
        t, y = sol.x, sol.y.T

        # Deactivate Kepler model
        self.model = 'full'

        # Check if solution has collided
        has_collide = self.check_collision(y[:, 0:3]*self.r_ad)
        sol.has_collide = has_collide

        return sol

    # This method solves transfer bvp with full model
    def solve_bvp(self, t0, y0):
        # Solve transfer
        sol = solve_bvp(self._dynamics,
                        self._bc,
                        t0,
                        y0,
                        verbose=0,
                        max_nodes=400,
                        tol=1e-4)
        t, y = sol.x, sol.y.T

        # Check if solution has collided
        has_collide = self.check_collision(y[:, 0:3]*self.r_ad)
        sol.has_collide = has_collide

        # Dimensionalize variables
        sol.x *= self.t_ad
        sol.y[0:3, :] *= self.r_ad
        sol.y[3:6, :] *= self.r_ad / self.t_ad

        return sol
