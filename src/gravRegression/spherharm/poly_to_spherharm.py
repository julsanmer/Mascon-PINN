import numpy as np
import math as mt

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList


# This computes spherical harmonics coefficients
# from polyhedron shape
def compute_CS(deg, rE, file_poly):
    # This initializes first-order terms
    def init_alphabeta(alpha, beta):
        # Initialize C11, S11
        alpha[0:2, 0:2, 1, 1] = np.array([[x3, x2],
                                          [x1, 0]]) / np.sqrt(3)
        beta[0:2, 0:2, 1, 1] = np.array([[y3, y2],
                                         [y1, 0]]) / np.sqrt(3)

        # Initialize C10
        alpha[0:2, 0:2, 1, 0] = np.array([[z3, z2],
                                          [z1, 0]]) / np.sqrt(3)

        return alpha, beta

    # This sets sectorial recursive trinomial
    def sectorial_trinomial():
        # Set trinomial matrices
        cte = (2*n - 1) / np.sqrt(2*n * (2*n + 1))
        sect_x = cte * np.array([[x3, x2],
                                 [x1, 0]])
        sect_y = cte * np.array([[y3, y2],
                                 [y1, 0]])

        return sect_x, sect_y

    # This sets subdiagonal recursive trinomial
    def subdiagonal_trinomial():
        # Set subdiagonal matrices
        cte = (2*n-1) / np.sqrt(2*n+1)
        subd_z = cte * np.array([[z3, z2],
                                 [z1, 0]])

        return subd_z

    # This sets vertical recursive trinomials
    def vertical_trinomial():
        # Set vertical matrices
        cte1 = (2*n-1) * np.sqrt((2*n-1)
                                   / ((2*n+1) * (n+m) * (n-m)))
        cte2 = -np.sqrt(((2*n-3) * (n+m-1) * (n-m-1))
                        / ((2*n+1) * (n+m) * (n-m)))
        vert_z = cte1 * np.array([[z3, z2],
                                  [z1, 0]])
        vert_r = cte2 * np.array(
            [[x3**2 + y3**2 + z3**2, 2*(x3*x2 + y3*y2 + z3*z2), x2**2 + y2**2 + z2**2],
             [2*(x3*x1 + y3*y1 + z3*z1), 2*(x2*x1 + y2*y1 + z2*z1), 0],
             [x1**2 + y1**2 + z1**2, 0, 0]])

        return vert_z, vert_r

    # This convolutes the kernel around A matrix
    def convolute(A, kernel):
        # Set output matrix and get dimensions
        B = np.zeros((n+1, n+1))
        dim_A, _ = A.shape
        dim_ker, _ = kernel.shape

        # Loop through A rows
        for i in range(dim_A):
            # Loop through A columns
            for j in range(dim_A-i):
                B[i:i+dim_ker, j:j + dim_ker] += A[i, j] * kernel

        return B

    # This computes sum(i!j!k!d_ijk)
    def sum_alphabeta(alpha_n, beta_n):
        # Initialize summations
        sum_alpha = 0
        sum_beta = 0

        # Loop through rows
        for i in range(n+1):
            # Decrease z degree
            deg_x = 0
            deg_y = i
            deg_z = n - i

            # Loop through columns
            for j in range(n+1-i):
                # Factorial i!j!k!
                fact = mt.factorial(deg_x) \
                       * mt.factorial(deg_y) \
                       * mt.factorial(deg_z)

                # Add term
                sum_alpha += fact * alpha_n[i, j]
                sum_beta += fact * beta_n[i, j]

                # Add degree on x and decrease z
                deg_x += 1
                deg_z -= 1

        return sum_alpha, sum_beta

    # Load polyhedron shape
    vert_list, face_list, n_vert, n_face =\
        loadPolyFromFileToList(file_poly)
    xyz_vert = np.array(vert_list)
    order_face = np.array(face_list)

    # Adimensionalize xyz_vert and initialize volume
    xyz_vert /= rE
    vol = 0

    # Preallocate spherical harmonics
    C = np.zeros((deg+1, deg+1))
    S = np.zeros((deg+1, deg+1))

    # Output status
    print('------- Initiating poly to SH -------')

    # Loop through faces
    for k in range(n_face):
        # Get vertexes of current face
        order = order_face[k, 0:3] - 1
        xyz1 = xyz_vert[order[0], 0:3]
        xyz2 = xyz_vert[order[1], 0:3]
        xyz3 = xyz_vert[order[2], 0:3]

        # Add volume term
        vol += abs(np.dot(np.cross(xyz1, xyz2), xyz3)) / 6

        # Extract vertexes
        x1, y1, z1 = xyz1[0], xyz1[1], xyz1[2]
        x2, y2, z2 = xyz2[0], xyz2[1], xyz2[2]
        x3, y3, z3 = xyz3[0], xyz3[1], xyz3[2]
        J = np.array([[x1, x2, x3],
                      [y1, y2, y3],
                      [z1, z2, z3]])
        detJ = np.linalg.det(J) * rE**3

        # Reset alpha and beta
        alpha = np.zeros((deg+1, deg+1, deg+1, deg+1))
        beta = np.zeros((deg+1, deg+1, deg+1, deg+1))

        # Initialize alpha and beta
        alpha, beta = init_alphabeta(alpha, beta)

        # Add first-order
        n = 1
        sum_alpha, sum_beta = sum_alphabeta(alpha[0:2, 0:2, 1, 0],
                                            beta[0:2, 0:2, 1, 0])
        C[1, 0] += detJ * sum_alpha / mt.factorial(4)
        S[1, 0] += detJ * sum_beta / mt.factorial(4)
        sum_alpha, sum_beta = sum_alphabeta(alpha[0:2, 0:2, 1, 1],
                                            beta[0:2, 0:2, 1, 1])
        C[1, 1] += detJ * sum_alpha / mt.factorial(4)
        S[1, 1] += detJ * sum_beta / mt.factorial(4)

        # Loop through degree
        for n in range(2, deg+1):
            # Loop through order
            for m in range(0, n+1):
                # Vertical term
                if m <= n - 2:
                    # Obtain vertical trinomials
                    vert_z, vert_r = vertical_trinomial()

                    # Obtain (n-1, m) alpha and beta
                    alpha_prev = alpha[0:n, 0:n, n-1, m]
                    beta_prev = beta[0:n, 0:n, n-1, m]

                    # Obtain (n-2, m) alpha and beta
                    alpha_prev2 = alpha[0:n-1, 0:n-1, n-2, m]
                    beta_prev2 = beta[0:n-1, 0:n-1, n-2, m]

                    # Do convolution to advance alpha and beta
                    c_alpha1 = convolute(alpha_prev, vert_z)
                    s_beta1 = convolute(beta_prev, vert_z)
                    if n > 2:
                        c_alpha2 = convolute(alpha_prev2, vert_r)
                        s_beta2 = convolute(beta_prev2, vert_r)
                    else:
                        c_alpha2 = vert_r
                        s_beta2 = beta_prev2

                    # Set alpha and beta trinomials for (n,m)
                    alpha[0:n+1, 0:n+1, n, m] = c_alpha1 + c_alpha2
                    beta[0:n+1, 0:n+1, n, m] = s_beta1 + s_beta2
                # Subdiagonal term
                elif m == n-1:
                    # Obtain subdiagonal trinomial
                    subd_z = subdiagonal_trinomial()

                    # Retrieve previous sectorial
                    alpha_prev = alpha[0:n, 0:n, n-1, n-1]
                    beta_prev = beta[0:n, 0:n, n-1, n-1]

                    # Do convolution to advance alpha and beta
                    c_alpha = convolute(alpha_prev, subd_z)
                    s_beta = convolute(beta_prev, subd_z)

                    # Set alpha and beta trinomials for (n,n-1)
                    alpha[0:n+1, 0:n+1, n, m] = c_alpha
                    beta[0:n+1, 0:n+1, n, m] = s_beta
                # Sectorial term
                elif m == n:
                    # Obtain sectorial trinomial (kernel)
                    sect_x, sect_y = sectorial_trinomial()

                    # Retrieve previous sectorial
                    alpha_prev = alpha[0:n, 0:n, n-1, m-1]
                    beta_prev = beta[0:n, 0:n, n-1, m-1]

                    # Do convolution to advance alpha and beta
                    c_alpha = convolute(alpha_prev, sect_x)
                    s_alpha = convolute(beta_prev, -sect_y)
                    c_beta = convolute(alpha_prev, sect_y)
                    s_beta = convolute(beta_prev, sect_x)

                    # Set alpha and beta trinomials for (n,n)
                    alpha[0:n+1, 0:n+1, n, m] = c_alpha + s_alpha
                    beta[0:n+1, 0:n+1, n, m] = c_beta + s_beta

                # Compute sum of alpha and beta
                sum_alpha, sum_beta = sum_alphabeta(alpha[0:n+1, 0:n+1, n, m],
                                                    beta[0:n+1, 0:n+1, n, m])

                # Get coefficients per face
                C_face = detJ * sum_alpha / mt.factorial(n+3)
                S_face = detJ * sum_beta / mt.factorial(n+3)

                # Add facet to C and S
                C[n, m] += C_face
                S[n, m] += S_face

        # Show progress
        if k % (int(n_face*0.05)) == 0:
            print('{:.2f}'.format(k/n_face*100)
                  + ' % faces processed')

    # Output status
    print('------- Finished poly to SH -------')

    # Add volume term
    vol *= pow(rE, 3)
    C /= vol
    S /= vol
    C[0, 0] = 1

    return C, S