import numpy as np
import pandas as pd
from generator import generate_data, DeePC, Hankel,UnionFind
from numpy.linalg import svd, matrix_rank, eigvalsh
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import statistics
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from functions import functions
from multiprocessing.pool import ThreadPool
import networkx as nx
from scipy.linalg import svd
from angle import friedrichs_angle
if __name__ == "__main__":
    np.random.seed(1)
    # ---- parameters ----
    v=2 # number of units in the interconnected graph
    n = 4          # state dimension
    p = 8          # observation dimension
    T_past = 10    # number of past steps to reconstruct
    T_fut = 20     # number of future observed steps
    T = T_past + T_fut
    T_total = T + 50  # total length of simulated trajectory

    # build a stable random A and observation C
    A = np.random.randn(n, n) * 0.5
    eigvals = np.linalg.eigvals(A)
    rho = max(abs(eigvals))
    if rho >= 1.0:
        A = A / (1.1 * rho)
    C = np.random.randn(p, n)

    Q = 0.01 * np.eye(n)   # process noise covariance
    R = 0.05 * np.eye(p)   # measurement noise covariance

    # Generate data to construct Hankel matrices (not used in this example)
    def generate_data(A, C, Q, R, T_total):
        n = A.shape[0]
        p = C.shape[0]
        z_data = np.zeros((T_total+1, n))
        y_data = np.zeros((T_total, p))
        z_data[0] = np.random.randn(n)
        for t in range(T_total):
            y_data[t] = C @ z_data[t] + np.random.multivariate_normal(np.zeros(p), R)
            z_data[t+1] = A @ z_data[t] + np.random.multivariate_normal(np.zeros(n), Q)
        return z_data, y_data

    z_data, y_data = generate_data(A, C, Q, R, T_total)

    # hankel matrix is of size (L*p, N)
    L = T_past + T_fut
    N = T_total - L + 1
    def construct_hankel(data, L):
        # Create a Hankel matrix from the data
        # The first L rows correspond to the past observations
        # The last row corresponds to the future observations
        H = np.zeros((L * p, N))
        for i in range(L):
            H[i * p:(i + 1) * p, :] = data[i:i + N, :].T
        return H
    H_y = construct_hankel(y_data, L)


    # simulate a observed trajectory
    z_true = np.zeros((T+1, n))
    y = np.zeros((T, p))
    z_true[0] = np.random.randn(n)

    u = np.random.multivariate_normal(np.zeros(n), Q, size=T)
    u= np.zeros((T,n))
    for t in range(T):
        y[t] = C @ z_true[t] + np.random.multivariate_normal(np.zeros(p), R)
        # u[t] = np.random.multivariate_normal(np.zeros(n), Q)
        z_true[t+1] = A @ z_true[t] + u[t]
    # y = (C @ z_true.T).T + np.random.multivariate_normal(np.zeros(p), R, size=T)

    print(z_true.shape)
    print(y.shape)

    # build mask for future observations (random subset each future time)
    # obs_frac = 0.5
    # mask = (np.random.rand(T_fut, p) < obs_frac)
    y_obs = y[-T_fut:, :]
    print(y_obs.shape)


    def observability_matrix(A, C, L):
        """Construct the extended observability matrix O_L with L block rows."""
        obs_matrix = C
        Ak = A
        for _ in range(1, L):
            obs_matrix = np.vstack((obs_matrix, C @ Ak))
            Ak = Ak @ A
        return obs_matrix
    
    def check_observability(A, C):
        
        n = A.shape[0]
        observability_matrix = C
        for i in range(1, n):
            observability_matrix = np.vstack((observability_matrix, C @ np.linalg.matrix_power(A, i)))
        
        rank_of_observability_matrix = np.linalg.matrix_rank(observability_matrix)
        is_observable = (rank_of_observability_matrix == n)
        
        return observability_matrix, is_observable

    observability_matrix, is_observable = check_observability(A, C)
    print("Is the system observable?",is_observable)

#         g = cp.Variable((T-L+1,1))
#     w_f = cp.Variable((N*q_central,1))
#     objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))
#                 # + lambda_g*cp.norm(g, 1) 
#     #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\                  
#     constraints = [ h_total[:Tini*q_central,:] @ g == wini,
#                     h_total[Tini*q_central:,:] @ g == w_f
#                                 ]
#     # box constraint on inputs
#     w_f_reshaped = cp.reshape(w_f, (-1, N))
#     constraints += [
#         w_f_reshaped[:m_central, :] <= 0.5,
#         w_f_reshaped[:m_central, :] >= -0.5
#     ]
#     problem = cp.Problem(cp.Minimize(objective), constraints) 
#     solver_opts = {
#     'max_iter': 10000,
#     'verbose': True     # Enable verbose output to debug
# }
#     # problem.solve(solver = cp.OSQP,**solver_opts)
#     problem.solve(solver = cp.SCS,verbose=False)
#     print("status:", problem.status)

    Z = cp.Variable((T+1, n))  # T+1 states: z[0], z[1], ..., z[T]
    constraints = []
    
    # Add dynamics constraints
    for t in range(T):
        constraints.append(Z[t+1, :] == A @ Z[t, :])
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # regularize if needed
        Rinv = np.linalg.pinv(R)

    obj_terms = []
    # Fit future observations only
    for t in range(T_past, T):
        resid = C @ Z[t, :] - y[t, :]
        obj_terms.append(cp.quad_form(resid, np.eye(p)))

    objective = cp.Minimize(cp.sum(obj_terms))

    prob = cp.Problem(objective, constraints)

    # Solve with a QP-capable solver
    prob.solve(solver=cp.OSQP, verbose=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=100000)

    print("status:", prob.status)
    print('smoothing residual:', prob.value)
    Z_est = Z.value   # shape (T+1,n)
    y_est = (C @ Z_est[:-1, :].T).T  # shape (T,p) - exclude last state
    print("Total observation reconstruction error:", np.linalg.norm(y_est - y, ord='fro')/np.linalg.norm(y, ord='fro'))

    # Extract reconstructed past states
    z_past_est = Z_est[:T_past, :]
    y_past_est = (C @ z_past_est.T).T
    print("Past state error (vs ground truth):", np.linalg.norm(z_past_est - z_true[:T_past, :], ord='fro')/np.linalg.norm(z_true[:T_past, :], ord='fro'))
    print("Past observation reconstruction:", np.linalg.norm(y_past_est - y[:T_past, :], ord='fro'))
    
    z_fut_est = Z_est[T_past:T, :]
    y_fut_est = (C @ z_fut_est.T).T
    print("Future state error (vs ground truth):", np.linalg.norm(z_fut_est - z_true[T_past:T, :], ord='fro')/np.linalg.norm(z_true[T_past:T, :], ord='fro'))
    print("Future observation error (should equal sqrt(residual)):", np.linalg.norm(y_fut_est - y[T_past:T, :], ord='fro'))
    print("sqrt(smoothing residual):", np.sqrt(prob.value))
    
    # def FB_split(w_ref, Phi, tol=1e-8): # Alberto's algorithm
    #     # Initialize w, z, v
    #     # w=np.vstack((w_ini, w_ref ))
    #     w = np.zeros((q*L,1))
    #     # w = 10*np.random.rand(self.q*self.L,1)-5
    #     kron=np.diag( np.kron(np.eye(self.N),Phi) ).reshape(-1, 1) # a vector containing all diagonal elements
    #     k=0
    #     for ite in range(self.max_iter):
    #         z = 2 * np.vstack(( np.zeros((self.q*self.Tini,1)) ,kron * (w[-self.q*self.N:]-w_ref) )) #O(2n)

    #         # Compute vk+1
    #         w_proj=  w - self.alpha*z # O(n)
    #         w = self.proj_h @ w_proj # O(n^2)
    #         k+=1
    #         # # # print( 'norm',np.linalg.norm(w - w_prev))
    #         # if np.linalg.norm(w - w_prev) < tol:
    #         #     break
    #     return w
    

    # print("y_obs:")
    # print(y_obs)
    # print(y_obs.reshape(-1, 1))
    # # F=functions(T,T_past, T_fut, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter,w_f.value)
    # y_split=FB_split(y_obs.reshape(-1, 1), Rinv)


