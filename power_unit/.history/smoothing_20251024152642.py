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

    # build a stable random A and observation C
    A = np.random.randn(n, n) * 0.5
    eigvals = np.linalg.eigvals(A)
    rho = max(abs(eigvals))
    if rho >= 1.0:
        A = A / (1.1 * rho)
    C = np.random.randn(p, n)

    Q = 0.01 * np.eye(n)   # process noise covariance
    R = 0.05 * np.eye(p)   # measurement noise covariance

    # simulate a trajectory
    z_true = np.zeros((T+1, n))
    y = np.zeros((T, p))
    z_true[0] = np.random.randn(n)
    for t in range(T):
        y[t] = C @ z_true[t] + np.random.multivariate_normal(np.zeros(p), R)
        z_true[t+1] = A @ z_true[t] + np.random.multivariate_normal(np.zeros(n), Q)
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

    Z = cp.Variable((T, n))
    constraints = []
    for t in range(T-1):
        constraints.append(Z[t+1, :] == A @ Z[t, :])

    # Data-fit objective on future times. Use R^{-1} weighting if you have R:
    # If R is scalar multiple of I you can use scalar weight. We'll implement general R.
    # Precompute inverse or use Cholesky factor for numerical stability
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # regularize if needed
        Rinv = np.linalg.pinv(R)

    obj_terms = []
    for k, t in enumerate(range(T_past, T)):
        # C @ z_t is (p,) expression
        resid = C @ Z[t, :] - y[t, :]
        # scalarized quadratic: resid^T R^{-1} resid
        # CVXPY has quad_form
        obj_terms.append(cp.quad_form(resid, Rinv))

    objective = cp.Minimize(cp.sum(obj_terms))

    # Optional tie-breaker regularizer if you want uniqueness / stability:
    # alpha = 1e-3
    # objective = cp.Minimize(cp.sum(obj_terms) + 0.5*alpha*cp.sum_squares(Z))

    prob = cp.Problem(objective, constraints)

    # Solve with a QP-capable solver. OSQP is a good choice for QP.
    prob.solve(solver=cp.OSQP, verbose=True, eps_abs=1e-5, eps_rel=1e-5, max_iter=100000)

    print("status:", prob.status)
    Z_est = Z.value   # shape (T,n)
    y_est = (C @ Z_est.T).T  # shape (T,p)

    # Extract reconstructed past states
    z_past_est = Z_est[:T_past, :]

    # If you want reconstructed outputs:
    y_past_est = (C @ z_past_est.T).T   # shape (T_past, p)
