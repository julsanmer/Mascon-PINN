import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from Basilisk import __path__

bsk_path = __path__[0]


# This tries to embed an ellipsoid w.r.t.
# surface points
def solve_dual_problem(P, tolerance=2.5*1e-5):
    d, N = P.shape

    Q = np.zeros((d + 1, N))
    Q[0:d, :] = P[0:d, 0:N]
    Q[d, :] = np.ones(N)

    count = 1
    err = 1
    u = np.ones(N) / N  # 1st iteration

    while err > tolerance:
        X = Q @ np.diag(u) @ Q.T
        M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        count += 1
        err = np.linalg.norm(new_u - u)
        u = new_u
        print(err)

    U = np.diag(u)

    # Compute A matrix for the ellipse
    A = (1 / d) * np.linalg.inv(P @ U @ P.T - np.outer(P @ u, P @ u))

    # Compute center of the ellipse
    c = P @ u

    return A, c

def plot_ellipse_and_points(P, A, c):
    # Plot the points
    plt.scatter(P[0, :], P[1, :], color='blue', label='Points P')

    # Plot the ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    ellipse = Ellipse(xy=c, width=2/np.sqrt(eigenvalues[0]), height=2/np.sqrt(eigenvalues[1]), angle=angle,
                      edgecolor='red', facecolor='none', label='Ellipse')
    plt.gca().add_patch(ellipse)

    # Set plot labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.title('Ellipse and Points')
    plt.show()


def plot_ellipsoid_and_points(points, A, center):
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    print(eigenvectors)
    axes = 1/np.sqrt(eigenvalues)

    # Generate points on the ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)
    x = axes[0] * np.cos(u) * np.sin(v)
    y = axes[1] * np.sin(u) * np.sin(v)
    z = axes[2] * np.cos(v)
    print(axes*16*1e3)
    print(center*16*1e3)

    # Rotate
    x = eigenvectors[0, 0] * x + eigenvectors[0, 1] * y + eigenvectors[0, 2] * z
    y = eigenvectors[1, 0] * x + eigenvectors[1, 1] * y + eigenvectors[1, 2] * z
    z = eigenvectors[2, 0] * x + eigenvectors[2, 1] * y + eigenvectors[2, 2] * z

    # Apply transformation to the ellipsoid points
    ellipsoid_points = np.vstack((x.ravel(), y.ravel(), z.ravel()))

    # Translate the ellipsoid to the specified origin
    transformed_ellipsoid = ellipsoid_points + center[:, np.newaxis]

    # Plot the ellipsoid
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(transformed_ellipsoid[0, :].reshape(100, 100),
                    transformed_ellipsoid[1, :].reshape(100, 100),
                    transformed_ellipsoid[2, :].reshape(100, 100),
                    color='b', alpha=0.2, linewidth=0, antialiased=False, label='Ellipsoid')

    # Plot the 3D points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=10, label='Points')

    # Set plot labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.title('Ellipsoid and 3D Points')
    plt.show()


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Set polyhedron file
poly_file = bsk_path + '/supportData/LocalGravData/EROS856Vert1708Fac.txt'
vert_list, face_list, n_vert, n_face = loadPolyFromFileToList(poly_file)
xyz_vert = np.array(vert_list) / (16*1e3)

# Example usage
# Generate random data points
np.random.seed(0)
P = xyz_vert.T

# Solve the dual problem
A, c = solve_dual_problem(P)

print("A matrix for the ellipse:")
print(A)
print("Center of the ellipse:")
print(c)

# Example usage
# Assuming you have already computed A and c using the solve_dual_problem function
#plot_ellipse_and_points(P, A, c)
plot_ellipsoid_and_points(P.T, A, c)
