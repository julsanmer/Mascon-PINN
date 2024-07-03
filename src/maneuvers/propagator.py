import numpy as np
import pickle as pck

from scipy.integrate import solve_ivp

from Basilisk.utilities import simIncludeGravBody


class Propagator:
    def __init__(self, asteroid, n_eval=100):
        # Evaluation points
        self.n_eval = n_eval

        # Tolerances
        self.rtol = 1e-6
        self.atol = 1e-6

        # Asteroid object
        self.asteroid = asteroid

    # This method propagates
    def propagate(self, t_span, pos0, vel0):
        # Stack initial position and velocity
        y0 = np.concatenate((pos0,
                             vel0))

        # Set timespan tuple and evaluation points
        t0, tf = t_span[0], t_span[1]
        t_eval = np.linspace(t0, tf, self.n_eval)

        # Integrate the dynamics
        prop_sol = solve_ivp(fun=self._dynamics,
                             t_span=t_span,
                             y0=y0,
                             method='DOP853',
                             atol=self.atol,
                             rtol=self.rtol,
                             t_eval=t_eval,
                             dense_output=True)

        # Extract times, positions, and velocities from the solution
        t = prop_sol.t
        pos = prop_sol.y[0:3, :].T
        vel = prop_sol.y[3:6, :].T

        return t, pos, vel

    def _dynamics(self, t, y):
        # Retrieve position and velocity
        pos, vel = y[0:3], y[3:6]

        # Asteroid angular velocity
        angvel = np.array([0,
                           0,
                           2*np.pi / self.asteroid.rot_period])

        # Compute gravity
        acc = self.asteroid.compute_gravity(pos)

        # Dynamics derivatives
        pos_dot = vel
        vel_dot = - 2*np.cross(angvel, vel) \
                  - np.cross(angvel, np.cross(angvel, pos)) \
                  + acc
        y_dot = np.concatenate((pos_dot,
                                vel_dot))

        return y_dot
