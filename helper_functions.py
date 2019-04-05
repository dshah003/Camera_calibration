import numpy as np
import os
import cv2
import glob
import sys, argparse
from scipy import optimize as opt

def normalize_matrix(points):
    points = points.astype(np.float64)
    x_mean, y_mean = np.mean(points, axis = 0)
    var_x, var_y = np.var(points, axis=0)
    s_x, s_y
    
def normalization_matrix(pts):
    pts = pts.astype(np.float64)
    x_mean, y_mean = np.mean(pts, axis =0)
    var_x, var_y = np.var(pts, axis=0)
    s_x = np.sqrt(2/var_x) 
    s_y = np.sqrt(2/var_y)
#     print("Matrix: {4} : meanx {0}, meany {1}, varx {2}, vary {3}, sx {5}, sy {6} ".format(x_mean, y_mean, var_x, var_y, name, s_x, s_y))
    n = np.array([[s_x, 0, -s_x*x_mean], [0, s_y, -s_y*y_mean], [0, 0, 1]])
    # print(n)
    n_inv = np.array([ [1./s_x ,  0 , x_mean], [0, 1./s_y, y_mean] , [0, 0, 1] ])
    return n.astype(np.float64), n_inv.astype(np.float64)

def getNormalizedCorrespondences(correspondences):
    normalized_correspondences = []
    views = len(correspondences)
    for i in xrange(views):
        imp, worldp = correspondences[i]
        imp = imp.reshape(len(imp), 2)
        N_x, N_x_inv = normalization_matrix(worldp)
        N_u, N_u_inv = normalization_matrix(imp)
        
        hom_imp = np.array([ [[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([ [[each[0]], [each[1]], [1.0]] for each in worldp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp
        
        for i in range(normalized_hom_objp.shape[0]):
        # 54 points. iterate one by onea
        # all points are homogeneous
            n_o = np.matmul(N_x,normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o/n_o[-1]
            
            n_u = np.matmul(N_u,normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u/n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:,:-1]        
        normalized_imp = normalized_imp[:,:-1]

        normalized_correspondences.append((imp, worldp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))
    return normalized_correspondences


def compute_Homography(correspondences):
    image_points = correspondences[0]
    object_points = correspondences[1]
    normalized_image_points = correspondences[2]
    normalized_object_points = correspondences[3]
    N_u = correspondences[4]
    N_x = correspondences[5]
    N_u_inv = correspondences[6]
    N_x_inv = correspondences[7]

    N = len(image_points)
    # print("Number of points in current view : ", N)

    M = np.zeros((2*N, 9), dtype=np.float64)
    # print("Shape of Matrix M : ", M.shape)

    # print("N_model\n", N_x)
    # print("N_observed\n", N_u)

    # create row wise allotment for each 0-2i rows
    # that means 2 rows.. 
    for i in xrange(N):
        X, Y = normalized_object_points[i] #A
        u, v = normalized_image_points[i] #B
        row_1 = np.array([ -X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
        row_2 = np.array([ 0, 0, 0, -X, -Y, -1, X*v, Y*v, v])
        M[2*i] = row_1
        M[(2*i) + 1] = row_2
    u, s, vh = np.linalg.svd(M)
    # print("Computing SVD of M")

    h_norm = vh[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)
    # print("Normalized Homography Matrix : \n" , h_norm)
    # print(N_u_inv)
    # print(N_x)
    # h = h_norm
    h = np.matmul(np.matmul(N_u_inv,h_norm), N_x)
    # if abs(h[2, 2]) > 10e-8:
    h = h[:,:]/h[2, 2]
    # print("Homography for View : \n", h )
    return h

def minimizer(initial_guess, X, Y, h, N):
    x_j = X.reshape(N, 2)
    projected = [0 for i in xrange(2*N)]
    for j in range(N):
        x, y = x_j[j]
        w = h[6]*x + h[7]*y + h[8]
        projected[2*j] = (h[0] * x + h[1] * y + h[2]) / w
        projected[2*j + 1] = (h[3] * x + h[4] * y + h[5]) / w
    # return projected
    return (np.abs(projected - Y))**2

def jacobian_cal(initial_guess, X, Y, h, N):
    x_j = X.reshape(N, 2)
    jacobian = np.zeros( (2*N, 9) , np.float64)
    for j in range(N):
        x, y = x_j[j]
        sx = np.float64(h[0]*x + h[1]*y + h[2])
        sy = np.float64(h[3]*x + h[4]*y + h[5])
        w = np.float64(h[6]*x + h[7]*y + h[8])
        jacobian[2*j] = np.array([x/w, y/w, 1/w, 0, 0, 0, -sx*x/w**2, -sx*y/w**2, -sx/w**2])
        jacobian[2*j + 1] = np.array([0, 0, 0, x/w, y/w, 1/w, -sy*x/w**2, -sy*y/w**2, -sy/w**2])
    return jacobian

def refine_homography(H, correspondences, corresp):
    image_points = corresp[0]
    object_points = corresp[1]
    normalized_image_points = corresp[2]
    normalized_object_points = corresp[3]
    N_u = corresp[4]
    N_x = corresp[5]
    N_u_inv = corresp[6]
    N_x_inv = corresp[7]
    N = normalized_object_points.shape[0]
    X = object_points.flatten()
    Y = image_points.flatten()
    h = H.flatten()
    h_prime = opt.least_squares(fun=minimizer, x0=h, jac=jacobian_cal, method="lm" , args=[X, Y, h, N], verbose=0)
    if h_prime.success:
        H =  h_prime.x.reshape(3, 3)
    H = H/H[2, 2]
    return H

def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2*M, 6), np.float64)

    def v_pq(p, q, H):
        v = np.array([
                H[0, p]*H[0, q],
                H[0, p]*H[1, q] + H[1, p]*H[0, q],
                H[1, p]*H[1, q],
                H[2, p]*H[0, q] + H[0, p]*H[2, q],
                H[2, p]*H[1, q] + H[1, p]*H[2, q],
                H[2, p]*H[2, q]
            ])
        return v

    for i in range(M):
        H = H_r[i]
        V[2*i] = v_pq(p=0, q=1, H=H)
        V[2*i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    # solve V.b = 0
    u, s, vh = np.linalg.svd(V)
    # print(u, "\n", s, "\n", vh)
    b = vh[np.argmin(s)]
    # print("V.b = 0 Solution : ", b.shape)

    # according to zhangs method
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    # print([vc,
    #         l,
    #         alpha,
    #         beta,
    #         gamma,
    #     uc])

    A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
    print("Intrinsic Camera Matrix is :")
    print(A)
    return A