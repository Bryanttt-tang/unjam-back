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
    Tini = 10    # number of past steps to reconstruct
    Tf = 20     # number of future observed steps
    T = Tini + Tf

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
    z_true = np.zeros((n, T))
    z_true[0] = np.random.randn(n)
    for t in range(T-1):
        z_true[:, t+1] = A @ z_true[:, t] + np.random.multivariate_normal(np.zeros(n), Q)
    y = (C @ z_true)+ np.random.multivariate_normal(np.zeros(p), R, size=T)

    print("True states:\n", z_true)
    print("Observations:\n", y)
    


  