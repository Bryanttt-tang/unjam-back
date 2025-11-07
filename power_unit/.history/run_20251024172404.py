import numpy as np
import pandas as pd
from generator import generate_data, DeePC, Hankel,UnionFind
from numpy.linalg import matrix_rank, eigvalsh
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import statistics
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from functions import functions
from multiprocessing.pool import ThreadPool
import networkx as nx
from scipy.linalg import svd,block_diag
from angle import friedrichs_angle
if __name__ == "__main__":
    np.random.seed(1)
    v = 5 # number of units in the interconnected graph
    n_subsystems = v
    G = nx.path_graph(n_subsystems)
    # Time step
    d_t = 0.01

    # Initialize lists to store matrices A, B, C, D for each subsystem
    A_list = []
    B_list = []
    C_list = []
    D_list = []

    # Generate random values for m_i (inertia), d_i (damping), k_ij (coupling)
    # mi = np.random.uniform(0.5, 1.0, n_subsystems)  # Inertia between [0.5, 1]
    mi = 2 # Inertia between [0.5, 1]
    d = np.random.uniform(0.5, 1, n_subsystems)    # Damping between [0, 10]
    k = np.random.uniform(1, 1.5, (n_subsystems, n_subsystems))  # Coupling between neighbors

    # Ensure no self-coupling (diagonal is 0)
    # np.fill_diagonal(K, 0)

    # Generate system matrices for each subsystem
    for i in range(n_subsystems):
        # Compute the coupling constant K_i as the sum of k_ij from neighbors in the graph
        neighbors = list(G.neighbors(i))  # Get neighbors of subsystem i
        K_i = np.sum([k[i, j] for j in neighbors])  # Sum of k_ij for neighbors

        # Define the A_i matrix
        A_i = np.array([[1, d_t],
                        [-K_i / mi * d_t, 1 - (d[i] / mi) * d_t]])
        
        # Define the B_i matrix
        B_i = np.array([[1],
                        [1]])
        
        # Define the C_i matrix based on neighbors' coupling constants
        if neighbors:
            C_i_value = np.sum([k[i, j] / mi for j in neighbors]) * d_t
        else:
            C_i_value = 0  # No neighbors case (edge node in the graph)
        
        C_i = np.array([[C_i_value, 0]])
        
        # Define the D_i matrix (zero)
        D_i = np.zeros((1, 1))
        
        # Append matrices to respective lists
        A_list.append(A_i)
        B_list.append(B_i)
        C_list.append(C_i)
        D_list.append(D_i)


    # def friedrichs_angle(A, B, tol=1e-10):
    #     """
    #     Compute the cosine of the Friedrichs angle between two subspaces
    #     spanned by the columns of A and B.
    #     """
    #     # Orthonormal bases
    #     QA, _ = np.linalg.qr(A)
    #     QB, _ = np.linalg.qr(B)
        
    #     # Project out intersection
    #     # Compute intersection basis
    #     U, s, Vh = svd(QA.T @ QB)
    #     # Indices where singular values are close to 1 (intersection)
    #     intersection = np.where(np.abs(s) > 1 - tol)[0]
    #     # Remove intersection directions
    #     QA_proj = QA
    #     QB_proj = QB
    #     if intersection.size > 0:
    #         QA_proj = QA[:, intersection.size:]
    #         QB_proj = QB[:, intersection.size:]
        
    #     # Compute singular values of the cross-Gram matrix
    #     M = QA_proj.T @ QB_proj
    #     svals = svd(M, compute_uv=False)
    #     # Friedrichs angle is the largest singular value
    #     return np.max(np.abs(svals))

    def observability_matrix(A, C, L):
        """Construct the extended observability matrix O_L with L block rows."""
        obs_matrix = C
        Ak = A
        for _ in range(1, L):
            obs_matrix = np.vstack((obs_matrix, C @ Ak))
            Ak = Ak @ A
        return obs_matrix

    def convolution_matrix(h, L):
        """Construct the convolution matrix C_L using the Markov parameters."""
        # Each row i has h(i), h(i-1), ..., h(0), padded with zeros
        rows = []
        for i in range(L):
            row_blocks = [h[j] if j <= i else np.zeros_like(h[0]) for j in range(i, -1, -1)]
            row_blocks += [np.zeros_like(h[0])]*(L - len(row_blocks))  # Pad zeros to reach L blocks
            rows.append(np.hstack(row_blocks))
        return np.vstack(rows)
    
    # Markovsky matrix
    def markov_parameters(A, B, C, D, L):
        """Generate the Markov parameters h(0), h(1), ..., h(L-1) of the system."""
        h = [D]  # Start with h(0) = D
        Ak = np.eye(A.shape[0])  # A^0 = Identity
        for k in range(1, L):
            Ak = Ak @ A
            h.append(C @ Ak @ B)
        return h
    
    def markov_matrix(A, B, C, D, L):
        """Construct the Markov matrix M_L using A, B, C, D, and length L."""
        h = markov_parameters(A, B, C, D, L)
        OL = observability_matrix(A, C, L)
        print('shape of OL:',OL.shape)
        CL = convolution_matrix(h, L)
        print('shape of CL:',CL.shape)
        # Construct M_L, which has an identity block on the top-left
        n = A.shape[0]  # State dimension
        mL = B.shape[1] * L
        identity_block = np.eye(mL)
        zero_block = np.zeros((mL,n))
        # upper_block = np.hstack((identity_block, zero_block))
        upper_block = np.hstack((zero_block,identity_block))
        lower_block = np.hstack((OL, CL))
        
        ML = np.vstack((upper_block, lower_block))
        print('shape of ML:',ML.shape)
        return ML

    def check_rank(M, expected_rank):
        """Check the rank of matrix M and verify it against expected_rank."""
        rank = np.linalg.matrix_rank(M)
        print(f"Rank of M_L: {rank}")
        return rank == expected_rank

    # Define example system matrices A, B, C, D
    # Replace with actual system matrices for your specific case
    A = np.array([[1, 0.1],
                [-0.5,0.7]])
    # A = np.array([[1, 0.1],
    #             [-0.5,1]])
    B = np.array([[0],[1]])
    C = np.array([[0.25,0]])
    D = np.zeros(1)
    A_couple=np.zeros((v*2,v*2))
    # print(A_couple)
    # print(A_couple[[0,1],[2,-1]])
    A_couple[1,2]=0.25
    A_couple[3,0]=0.25
    A_aug=np.kron(np.eye(v),A)+A_couple
    B_aug=np.kron(np.eye(v),B)
    C_aug=np.kron(np.eye(v),C)
    D_aug=np.kron(np.eye(v),D)

    # Define desired L and expected rank
    Tini = 3*v+1 # length of the initial trajectory
    N = 5  # prediction horizon
    L=Tini+N

    mL = B_aug.shape[1] * L
    n = A_aug.shape[0]
    expected_rank = mL + n

    # Construct M_L and check its rank
    ML = markov_matrix(A_aug, B_aug, C_aug, D_aug, L)
    is_full_rank = check_rank(ML, expected_rank)
    print(f"M_L has the expected rank {expected_rank}: {is_full_rank}")

    # print('A_aug',A_aug)
    A_list = []
    B_list = []
    C_list = []
    for i in range(v):
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)

        # # Display the generated matrices for each subsystem
    for i in range(n_subsystems):
        print(f"Subsystem {i+1}:")
        print(f"A_{i+1} = \n{A_list[i]}")
        print(f"B_{i+1} = \n{B_list[i]}")
        print(f"C_{i+1} = \n{C_list[i]}")
        print(f"D_{i+1} = \n{D_list[i]}")

    eigenvalues, _ = np.linalg.eig(A_list[-1])
    # Check for stability (all eigenvalues should have magnitudes less than 1 for discrete-time system)
    is_stable = np.all(np.abs(eigenvalues) < 1)
    print("Eigenvalues:")
    print(eigenvalues)
    print("Is the system stable? --", is_stable)
    # print('The reference: \n',wref)
    def check_controllability(A, B):
        n = A.shape[0]
        controllability_matrix = B
        
        for i in range(1, n):
            controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
        
        rank_of_controllability_matrix = np.linalg.matrix_rank(controllability_matrix)
        
        is_controllable = rank_of_controllability_matrix == n
        
        return controllability_matrix, is_controllable
    def check_observability(A, C):
        
        n = A.shape[0]
        observability_matrix = C
        for i in range(1, n):
            observability_matrix = np.vstack((observability_matrix, C @ np.linalg.matrix_power(A, i)))
        
        rank_of_observability_matrix = np.linalg.matrix_rank(observability_matrix)
        is_observable = (rank_of_observability_matrix == n)
        
        return observability_matrix, is_observable

    controllability_matrix, is_controllable = check_controllability(A, B)
    observability_matrix, is_observable = check_observability(A, C)
    print("Is the system controllable?", is_controllable)
    print("Is the system observable?",is_observable)
    ''' Integer invariants '''

    n = np.shape(A)[0]  # dimesion of the state
    m = np.shape(B)[1]  # dimesion of the input
    p = np.shape(C)[0]  # dimesion of the output
    q = p+m             # dimesion of input/output pair
    
    def create_connected_graph(n):
        """
        Creates a connected graph with n nodes and n-1 edges (a tree).
        # """
        G = nx.path_graph(n)
        return G

    def add_random_edge(G):
        """
        Adds a random edge to the graph G using np.random.choice.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        max_edges = n * (n - 1) // 2  # Maximum number of edges in a complete graph

        if G.number_of_edges() == max_edges:
            print("Already a complete graph")
            return

        while True:
            u, v = np.random.choice(nodes, 2, replace=False)  # Randomly select two distinct nodes
            if not G.has_edge(u, v):  # Check if the edge does not already exist
                G.add_edge(u, v)  # Add the new edge to the graph
                break  # Exit the loop once a new edge is added
    def increase_edge_connectivity(G, target_connectivity):
        """
        Increase the edge connectivity of the graph G to the target_connectivity.
        """
        current_connectivity = nx.edge_connectivity(G)
        
        while current_connectivity < target_connectivity:
            add_random_edge(G)
            current_connectivity = nx.edge_connectivity(G)   
                 
    def find_connected_components(pairs):
        uf = UnionFind()

        # Add all nodes to the union-find structure
        for node1, node2 in pairs:
            uf.add(node1)
            uf.add(node2)
            uf.union(node1, node2)

        # Find the root for each node to determine the connected components
        components = {}
        for node in uf.parent:
            root = uf.find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)

        return list(components.values())
    # create random graph


    graph = create_connected_graph(v)   
    increase_edge_connectivity(graph, 3)

    # while graph.number_of_edges() <= (v * (v - 1)) // 2:
    e=2*len(graph.edges()) # directed graph  
    v_con=nx.node_connectivity(graph)
    e_con=nx.edge_connectivity(graph)
    print('vertex connectivity:',v_con)
    print('edge connectivity:',e_con)
    # add_random_edge(graph)
    # print("Initial number of edges:", graph.number_of_edges())
    # print("Is the graph connected?", nx.is_connected(graph))
    m_dis=v*m+e
    m_central=v*m
    p_dis=v*p         
    p_central=v*p
    q_dis=(m_dis+p_dis)
    q_central=(m_central+p_central)
    T = v*L+v*n+150   # number of data points
    R=0.1*np.eye(m_central)
    R_dis=0.1*np.eye(m_dis)
    R_dis[1,1]=0
    R_dis[3,3]=0
    Q=10*np.eye(p_dis)
    Phi=np.block([[R, np.zeros((R.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R.shape[1])), Q]])
    Phi_dis=np.block([[R_dis, np.zeros((R_dis.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R_dis.shape[1])), Q]])
    lambda_g = 1          # lambda parameter for L1 penalty norm on g
    lambda_1=1
    # r = np.ones((p,1)) # reference trajectory, try to give different defitions of this to see if DeePC works!


    ''' Generate data '''
    def increase_snr(y, snr_db):
        signal_variance = np.var(y)
        desired_snr_linear = 10 ** (snr_db / 10)
        noise_variance = signal_variance / desired_snr_linear
        noise = np.random.normal(0, np.sqrt(noise_variance), y.shape)
        y_noisy = y + noise
        return y_noisy
    SNR=[]
    Error=[]
    w_star=[]

    # for noise in np.arange(0.1,1.1,0.1):
    X0 = np.random.rand(n,v)  
    # X0=np.random.uniform(-100, 100, (n, v))#initial state of 10 units
    generator = generate_data(T,Tini,N,p,m,n,v,e,A_list,B_list,C_list,D_list,graph,0)
    xData, uData ,yData, uData_dis ,yData_dis, yData_noise = generator.generate_pastdata(X0)
    print(np.var(uData))

    # print('var in original',np.var(yData))
    print('max U:',np.max(uData))
    print('max Y:',np.max(yData))
    print('max X:',np.max(xData))
    # snr_linear = np.var(yData) / noise
    # snr_db = 10 * np.log10(snr_linear)
    # print('SNR',snr_db)
    # SNR.append(1/snr_linear)
    wData=np.vstack((uData,yData))
    wData_dis=np.vstack((uData_dis,yData_dis))
    wData_noise=np.vstack((uData,yData_noise))
    # print('uData:',uData[:,2])
    # print('yData:',yData[:,2])
    # print('uData_dis:',uData_dis[:,2])
    # print('wData:',wData[:,2])
    # print('wData_dis:',wData_dis[:,2])
    wini = wData[:, -Tini:].reshape(-1, 1, order='F')
    wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
    wini_noise = wData_noise[:, -Tini:].reshape(-1, 1, order='F')
    # print('wini\n',wini) # (Tini*q=3*(m+p))
    print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

    # random_vector=np.zeros((q_central, N))
    # random_vector_dis=np.zeros((q_dis, N))
    ref_dis = np.array([[1], [10], [1], [10]])
    print(np.tile(ref_dis, N))
    u_ref=0.25*np.ones((m_central,1))
    y_ref=0.25*np.ones((p_central,1))
    random_vector=np.vstack((0.25*np.ones((m_central,N)),0.25*np.ones((p_central, N)) ))
    random_vector_dis=np.vstack((0.25*np.ones((m_dis, N)),0.25*np.ones((p_dis, N)) ))
    # wref=np.tile(r,N).reshape(-1,1, order='F')
    wref=random_vector.reshape(-1,1, order='F')
    wref_dis=random_vector_dis.reshape(-1,1, order='F')
    print('wref.shape:',wref_dis.shape)

    # Get M matrix
    second_indices = []
    excluded_indices=[]
    base_index = 0
    excluded_indices.append(base_index)
    node_to_index = {node: v + 2 * len(graph.edges) + idx for idx, node in enumerate(graph.nodes())}
    for i in range(v-1):
        neighbors_ex = sorted(graph.neighbors(i))
        base_index += len(neighbors_ex)+1
        excluded_indices.append(base_index)
    first_indices = [i for i in np.arange(m_dis) if i not in excluded_indices]
    for i in range(v):
        neighbors = sorted(graph.neighbors(i))
        # print(neighbors)
        for j in neighbors:
                # Update second_indices using the mapping
            second_indices.append(node_to_index[j])
    pairs = np.transpose([first_indices, second_indices])
    # print('pairs: \n',pairs)
    # Initialize the kernel matrix M
    M = np.zeros(( len(pairs), q_dis))
    # Assign -1 and 1 to the corresponding places in M
    for i, (idx1, idx2) in enumerate(pairs):
        M[i, idx1] = -1
        M[i, idx2] = 1
    M_inv=np.linalg.inv(M@M.T)  
    P_ker=np.eye(L*q_dis)-np.kron(np.eye(L),M.T@M_inv@M)
    P_ker_sin=np.eye(q_dis)-M.T@M_inv@M
    # print('Inter:\n',np.eye(M.shape[1])-M.T@M_inv@M)
    connected_components = find_connected_components(pairs)
    # print('connected_components',connected_components)
    
    ''' Hankel matrix '''
    
    params_H = {"uData": uData,
                "yData": yData,
                "Tini": Tini,
                "N": N,
                "n":v*n,
            }
    H = Hankel(params_H)

    # params_H_noise = {"uData": uData,
    #             "yData": yData_noise,
    #             "Tini": Tini,
    #             "N": N,
    #             "n":v*n,
    #         }
    # H_noise = Hankel(params_H_noise)
    
    params_H_dis = {"uData": uData_dis,
                "yData": yData_dis,
                "Tini": Tini,
                "N": N,
                "n":v*n,
            }
    H_dis = Hankel(params_H_dis)

    H_j=[]
    hankel_dis=[]
    start_row = 0
    for j in range(v):
        len_nei=len(sorted(graph.neighbors(j)))
        num_rows = m + len_nei
        end_row = start_row + num_rows
        params_H1 = {"uData": uData_dis[start_row:end_row,:],
        "yData": yData_dis[j*p:(j+1)*p,:],
        "Tini": Tini,
        "N": N,
        "n":n,
        }
        H_j.append(Hankel(params_H1))
        hankel_dis.append(Hankel(params_H1).Hankel)
        start_row = end_row
        
    h_total=H.Hankel
    h_dis=H_dis.Hankel
    Up = H.Up
    Uf = H.Uf
    Yp = H.Yp
    Yf = H.Yf

    # Example usage:
    print('+++++')
    print('rank of P_ker:', np.linalg.matrix_rank(P_ker))


    # blocks = 12
    # # --- 1) nullspace of G (orthonormal basis U_p for ker(G)) ---
    # # SVD-based nullspace extraction
    # Ug, sg, Vtg = svd(, full_matrices=True)   # Vtg shape 6x6 (rows are V^T)
    # rankG = np.sum(sg > (1e-10 * max(G.shape)))   # numeric rank
    # rP = 6 - rankG
    # if rP == 0:
    #     raise ValueError("Ker(G) is trivial (rank(G)=6); P_kerG = 0, so D = {0}.")

    # U_p = Vtg.T[:, rankG:]   # shape 6 x rP ; columns are orthonormal basis of ker(G)

    # # --- 2) build B by kron(I_blocks, U_p) ---
    # B = np.kron(np.eye(blocks), U_p)   # shape 72 x (blocks * rP)

    # # --- 3) checks ---
    # print("U_p.shape =", U_p.shape)
    # print("B.shape =", B.shape)
    # print("expected d_D =", blocks * rP)
    # print("matrix_rank(B) =", matrix_rank(B, tol=1e-8))
    # # orthonormality check: B.T @ B should be close to identity
    # print("||B.T B - I||_F =", np.linalg.norm(B.T @ B - np.eye(B.shape[1]), ord='fro'))

    w, V = np.linalg.eigh(P_ker_sin)
    mask = w > 0.5
    U_p = V[:, mask]
    P_k = np.kron(np.eye(L), U_p)   # shape: 72 x (blocks * rP)
    # B should have orthonormal columns if U_p columns are orthonormal
    print("B.shape =", P_k.shape)
    print("||B.T @ B - I|| =", np.linalg.norm(P_k.T @ P_k - np.eye(P_k.shape[1]))) 
    print('rank of P_k:', np.linalg.matrix_rank(P_k))

    print('shape of h_dis:',h_dis.shape)
    print('rank of h_dis:', np.linalg.matrix_rank(h_dis))
    print('shape of P_ker:',P_ker.shape)
    print('rank of P_ker:', np.linalg.matrix_rank(P_ker))

    print(M)
    ker=np.kron(np.eye(L),M)
    print('shape of ker:',ker.shape)
    print('rank of ker:', np.linalg.matrix_rank(ker))

    # we use the list of hankel_dis to build the block diagonal hankel matrix
    H_list = []
    shapes = []
    ranks = []
    for hankel in hankel_dis:
        U, S, Vt = svd(hankel, full_matrices=True)
        r_v = matrix_rank(hankel)
        A = U[:, :r_v]
        H_list.append(A)
        shapes.append(hankel.shape[0])
        ranks.append(r_v)
    # Block diagonal
    H_prod = block_diag(*H_list)
    print('shape of block diagonal hankel:',H_prod.shape)

    cosine_angle = friedrichs_angle(H_prod , ker)
    print(f"The Friedrichs angle between the two subspaces is: {np.degrees(cosine_angle):.2f} degrees")
    print('q*L',L*q*v)
    print('shape of H',h_total.shape)
    # h_total=H_dis.Hankel
    # print(h_total.shape)
    h=[]
    for i in range(len(H_j)):
        h.append(H_j[i].Hankel)
    

    max_iter=500
    dis_iter=10
    alpha=0.1
    num_runs=1
    cost_data = np.zeros((num_runs, max_iter+1))
    cost_data1 = np.zeros((num_runs, max_iter+1))

   # Get the optimum w*
    g = cp.Variable((T-L+1,1))
    w_f = cp.Variable((N*q_central,1))
    objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))
                # + lambda_g*cp.norm(g, 1) 
    #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\                  
    constraints = [ h_total[:Tini*q_central,:] @ g == wini,
                    h_total[Tini*q_central:,:] @ g == w_f
                                ]
    # box constraint on inputs
    w_f_reshaped = cp.reshape(w_f, (-1, N))
    constraints += [
        w_f_reshaped[:m_central, :] <= 0.5,
        w_f_reshaped[:m_central, :] >= -0.5
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints) 
    solver_opts = {
    'max_iter': 10000,
    'verbose': True     # Enable verbose output to debug
}
    # problem.solve(solver = cp.OSQP,**solver_opts)
    problem.solve(solver = cp.SCS,verbose=False)

    for exp in range(num_runs):
        np.random.seed(exp)
        F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter,w_f.value)
        # F_noise=functions(T,Tini, N, v, e, m, 1, p, M, H_noise.Hankel, h, connected_components, graph, alpha, max_iter, dis_iter)
        # lqr_exp_time=[]
        # dis_lqr_exp_time=[]
        
        wini = wData[:, -Tini:].reshape(-1, 1, order='F')
        # print('wini\n',wini)
        wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
        start_markovski = time.process_time()
        # Markovski solution of data-driven control
        # W0, S, VT = np.linalg.svd(h_total[q_central*Tini:,:])
        W0=h_total[q_central*Tini:,:]
        print('m*Tf+n:',m_central*N+n*v)
        print('rank W0:',np.linalg.matrix_rank(W0))
        print(W0.shape)
        g_free=np.linalg.pinv(np.vstack((Up,Yp,Uf)))@np.vstack((wini, np.zeros((m_central*N,1)) ))
        y_free=Yf@g_free
        w_free=np.vstack((np.zeros((m_central,N)),y_free.reshape(-1,N))).reshape(-1, 1, order='F')
        # print('w_free \n',w_free)
        w_markov=W0@np.linalg.pinv(W0.T@W0)@W0.T@(wref-w_free)+w_free
        end_markovski = time.process_time()

        start_alberto = time.process_time()
        w_split = F.lqr(wini, wref, Phi)
        w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
        end_alberto = time.process_time()

        w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
        print('w_markov \n',w_markov)
        print('w_split \n',w_split[q_central*Tini:])
        print('CVX \n',w_f.value)
        print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:])/np.linalg.norm(w_f.value) )
        print('The differnce between CVX and Markovski methods: ',np.linalg.norm(w_f.value - w_markov)/np.linalg.norm(w_f.value) )
        # print('The differnce between distrbuted lqr and lqr methods: ',np.linalg.norm(w_split_dis - w_split[q_central*Tini:])/np.linalg.norm(w_markov) )
        # print('w_split-dis \n',w_split_dis)
        print(f"Running time of Markovski Algo: ",end_markovski-start_markovski)
        print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
        print(F.k_lqr)
        # cost_data[exp, :] = np.squeeze(F.E)
        # cost_data1[exp, :] = np.squeeze(F.E1)

    # mean_cost = np.mean(cost_data, axis=0)
    # std_cost = np.std(cost_data, axis=0)
    # mean_cost1 = np.mean(cost_data1, axis=0)
    # std_cost1 = np.std(cost_data1, axis=0)
    # # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(max_iter+1), mean_cost, color='red', label='Mean cost')
    # plt.fill_between(np.arange(max_iter+1), mean_cost - std_cost, mean_cost + std_cost, color='red', alpha=0.3)
    # plt.yscale('log')  # Use logarithmic scale if you want to show linear convergence clearly
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost vs Iterations (Convergence of LQT Algorithm)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(max_iter+1), mean_cost1, color='red', label='Mean cost')
    # plt.fill_between(np.arange(max_iter+1), mean_cost1 - std_cost1, mean_cost1 + std_cost1, color='red', alpha=0.3)

    # plt.yscale('log')  # Use logarithmic scale if you want to show linear convergence clearly
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost vs Iterations (Convergence of LQT Algorithm)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # df = pd.DataFrame({
    # 'Mean': mean_cost,
    # 'Std': std_cost,
    # 'Mean1': mean_cost1,
    # 'Std1': std_cost1
    #     })

    #     # Save DataFrame to CSV
    # df.to_csv('convergence-2.csv', index=False)
    # w_split_noise = F_noise.lqr(wini, wref, Phi)
    start_dist = time.process_time()
    w_split_dis = F.distributed_lqr(wini_dis, wref_dis,Phi_dis)
    end_dist = time.process_time()
    print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
    # print(f"Running time of worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)


    ## 2025-10-16: comparison between direct projection and alternating projection
    random_vector_dis=np.random.uniform(0, 1, size=(q_dis, L))
    w_ran_dis=random_vector_dis.reshape(-1,1, order='F')
    errors = []
    direct_projection=F.proj(h_dis,w_ran_dis)
    # direct_projection=h_total@np.linalg.pinv(h_total)@w_ran
    with ThreadPool(processes=8) as pool:
        for ite in tqdm(range(1,26)):
            projected_point_alternating = F.alternating_projections(F.proj_h_sub, w_ran_dis, pool, num_iterations=ite)
            error = np.linalg.norm(direct_projection-projected_point_alternating)
            errors.append(error)
    # plt.cla()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 26), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'Difference between direct Projection vs. Alternating Projection (System with {v} units)')
    plt.grid(True)
    # plt.pause(1) 
    plt.show()  

    ## 2025-10-16: comparison between direct projection and alternating projection for box constraint
    random_vector=np.random.uniform(0, 1, size=(q_central, L))
    w_ran=random_vector.reshape(-1,1, order='F')
    errors = []
    projected_point_alternating1 = F.alternating_projections2(F.proj_h_sub, w_ran, num_iterations=2)
    projected_point_alternating10 = F.alternating_projections2(F.proj_h_sub, w_ran, num_iterations=10)
    # print('projected_point_alternating1',projected_point_alternating1)
    # print('projected_point_alternating10',projected_point_alternating10)
    # print('norm of w_ran:',np.linalg.norm(w_ran))
    print('norm of projected_point_alternating1:',np.linalg.norm(projected_point_alternating1))
    print('difference between 1 and 10 iterations:',np.linalg.norm(projected_point_alternating1 - projected_point_alternating10)/np.linalg.norm(projected_point_alternating1))
   

    # U, S, VT = np.linalg.svd(h_total)
    # rank_total = np.linalg.matrix_rank(h_total)
    # U_truncated = U[:, :rank_total]
    # CVXPY
    start_cvx = time.time()
    g = cp.Variable((T-L+1,1))
    
    w_f = cp.Variable((N*q_central,1))
    objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))\
                + lambda_g*cp.norm(g, 1) 
    #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\
                    
    constraints = [ h_total[:Tini*q_central,:] @ g == wini,
                    h_total[Tini*q_central:,:] @ g == w_f
                                ]
    # box constraint on inputs
    w_f_reshaped = cp.reshape(w_f, (-1, N))
    constraints += [
        w_f_reshaped[:m_central, :] <= 0.5,
        w_f_reshaped[:m_central, :] >= 0
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints) 
    solver_opts = {
    'max_iter': 10000,
    'verbose': True     # Enable verbose output to debug
}
    # problem.solve(solver = cp.OSQP,**solver_opts)
    problem.solve(solver = cp.SCS,verbose=False)
    end_cvx = time.time()
    print('Running time of CVXPY for single LQR: ',end_cvx-start_cvx)

    # # MPC
    # x = cp.Variable((2*n, N+1))  # State trajectory
    # u = cp.Variable((m_central, N))    # Control input trajectory
    # y= cp.Variable((p_central, N))
    # # Objective function and constraints
    # cost = 0
    # constraints = [x[:, 0] == xData[:,-1]]  # Initial state constraint

    # for k in range(N):
    #     # Quadratic cost function for tracking
    #     cost += cp.quad_form(y[:, k:k+1] - y_ref, Q) + cp.quad_form(u[:, k:k+1]-u_ref, R)
        
    #     # System dynamics constraint
    #     constraints += [y[:, k] == C_aug @ x[:, k]]
    #     constraints += [x[:, k+1] == A_aug @ x[:, k] + B_aug @ u[:, k]]
    #     # # Input constraints
    #     # constraints += [cp.norm_inf(u[:, k]) <= u_max]
        
    #     # # State constraints
    #     # constraints += [cp.norm_inf(x[:, k]) <= x_max]

    # # Solve the optimization problem
    # problem_mpc = cp.Problem(cp.Minimize(cost), constraints)
    # problem_mpc.solve()
    # print('mpc solver \n',np.vstack((u.value, y.value)) )
    # print('The differnce between CVX and MPC methods: ',np.linalg.norm(w_f.value - np.vstack((u.value, y.value)).reshape(-1, 1, order='F'))/np.linalg.norm(w_f.value) )

    # diff=np.linalg.norm(w_split-np.vstack((wini,wref)) )
    print('error',F.E[-1])
    # print('last error',F.E1[-1])
    plt.plot(range(0, len(F.E)), np.squeeze(F.E))
    # plt.plot(range(0, len(F.E)), np.squeeze(F.E1))
    # plt.plot(range(0, max_iter+1), np.squeeze(F_noise.E))
    # # plt.plot(range(0, max_iter+1), np.squeeze(F.E_dis))
    # plt.ylabel('Error')
    plt.legend(['Total', 'last'])
    plt.title('Convergence Error of LQR')
    plt.grid(True)
    plt.pause(1)
    # plt.show(block=False)
    # plt.pause(0.001)
    # g_off=np.linalg.inv(U_truncated[:Tini*q_central,:])@wini
    print('iterations:',F.k_lqr)
    # w_f_off=U_truncated[Tini*q_central:,:]@np.linalg.pinv(U_truncated[:Tini*q_central,:])@wini
    w_f_off=h_total[Tini*q_central:,:]@np.linalg.pinv(h_total[:Tini*q_central,:])@wini
    print('The final cost of CVX:',problem.value)
    print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:])/np.linalg.norm(w_f.value) )
    # print('The differnce between offline CVX and Alberto methods: ',np.linalg.norm(w_f_off - w_split[q_central*Tini:]) )
    # print('The differnce between free and noisy Alberto methods: ',np.linalg.norm(w_split_noise[q_central*Tini:] - w_split[q_central*Tini:]) )
    # print('error',F.E[-1])
    # print('LQT tracking error: ',np.linalg.norm(F_noise.E[-1] -F.E[-1])/ np.linalg.norm(F.E[-1]))
    # Error.append(np.linalg.norm(F_noise.E[-1] -F.E[-1])/ np.linalg.norm(F.E[-1]))
    # w_star.append(np.linalg.norm(w_split_noise[q_central*Tini:] - w_split[q_central*Tini:])/np.linalg.norm(w_split[q_central*Tini:]))
    # print('The differnce between centralized and distributed methods: ',np.linalg.norm(w_split_dis[q_dis*Tini:] - w_split[q_dis*Tini:]) )
    
    # print(SNR)
    # plt.figure(figsize=(10, 6))
    # plt.plot(SNR, Error) 
    # plt.xlabel('1/SNR')
    # plt.ylabel('Error')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(SNR, w_star) 
    # plt.show()
    # plt.xlabel('1/SNR')
    # plt.ylabel('w*')
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.linspace(0, Tsim,Tsim), usim[0,:], label='CVXPY', color='blue')
        # plt.plot(np.linspace(0, Tsim,Tsim), usim2[0,:], label='lqr', color='red')
        # plt.plot(np.linspace(0, Tsim,Tsim), usim3[0,:], label='dis_lqr', color='black')
        # plt.ylabel('u')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    Tsim=101  
    solver='CVXPY'
    params_D = {'H': H, # an object of Hankel
                'H_dis':H_dis,
                'h_dis':h,
                'ML':ML,
                'xData':xData,
                'Phi':Phi,
                'Phi_dis':Phi_dis,
                'Q':Q,
                'R':R,
                'T':T,
                'Tini':Tini,
                'N':N,
                'n':n,
                'v':v,
                'e':e,
                'm':m,
                'p':p,
                'M':M,
                'connected_components':connected_components,
                'graph':graph,
                'alpha':alpha,
                'max_iter':max_iter,
                'dis_iter':dis_iter,
                'wref_dis' : wref_dis,
                'wref' : wref,
                'w_star':w_f.value,
                'A_aug':A_aug,
                'B_aug':B_aug,
                'C_aug':C_aug}
    Xsim=[]
    Usim=[]
    Ysim=[]
    Xsim2=[]
    Usim2=[]
    Ysim2=[]
    Xsim3=[]
    Usim3=[]
    Ysim3=[]
    Xsim4=[]
    Usim4=[]
    Ysim4=[]
    for exp in range(1,2):
        deepc = DeePC(params_D,'CVXPY','Hankel',exp)   
        x0 =  np.copy(xData[:, -(exp+1)])
        print('x0:',x0)
        start_deepc=time.process_time()
        xsim, usim, ysim = deepc.loop(Tsim,A_list,B_list,C_list,D_list,x0)
        end_deepc=time.process_time()
        Xsim.append(xsim)
        Usim.append(usim)
        Ysim.append(ysim)
        print('Total DeepC running time: ', end_deepc-start_deepc)

        deepc2 = DeePC(params_D,'dis_lqr','Hankel',exp)   
        x0 =  np.copy(xData[:, -(exp+1)])
        print('x0:',x0)
        start_deepc2=time.process_time()
        xsim2, usim2, ysim2 = deepc2.loop(Tsim,A_list,B_list,C_list,D_list,x0)
        end_deepc2=time.process_time()
        Xsim2.append(xsim2)
        Usim2.append(usim2)
        Ysim2.append(ysim2)
        print('Total Alberto running time: ', end_deepc2-start_deepc2)

        # deepc3 = DeePC(params_D,'lqr','Markov',exp)   
        # start_deepc3=time.process_time()
        # xsim3, usim3, ysim3 = deepc3.loop(Tsim,A_list,B_list,C_list,D_list,x0)
        # end_deepc3=time.process_time()
        # Xsim3.append(xsim3)
        # Usim3.append(usim3)
        # Ysim3.append(ysim3)
        # print('Total Markov running time: ', end_deepc3-start_deepc3)

        
        # deepc4 = DeePC(params_D,'mpc','Hankel',exp)   
        # start_deepc4=time.process_time()
        # xsim4, usim4, ysim4 = deepc4.loop(Tsim,A_list,B_list,C_list,D_list,x0)
        # end_deepc4=time.process_time()
        # Xsim4.append(xsim4)
        # Usim4.append(usim4)
        # Ysim4.append(ysim4)
        # print('Total mpc running time: ', end_deepc3-start_deepc3)

    usim_results = np.array(Usim)  # Shape (5, control_dim, Tsim)
    xsim_results = np.array(Xsim)  # Shape (5, state_dim, Tsim)
    ysim_results = np.array(Ysim)
    usim_results2 = np.array(Usim2)  # Shape (5, control_dim, Tsim)
    xsim_results2 = np.array(Xsim2)  # Shape (5, state_dim, Tsim)
    ysim_results2 = np.array(Ysim2)
    # usim_results3 = np.array(Usim3)  # Shape (5, control_dim, Tsim)
    # xsim_results3 = np.array(Xsim3)  # Shape (5, state_dim, Tsim)
    # ysim_results3 = np.array(Ysim3)
    # usim_results4 = np.array(Usim4)  # Shape (5, control_dim, Tsim)
    # xsim_results4 = np.array(Xsim4)  # Shape (5, state_dim, Tsim)
    # ysim_results4 = np.array(Ysim4)
    print('shape of Xsim:',xsim_results.shape)

    # x0 =  np.copy(xData[:, -1])
    # print(x0)
    # deepc2 = DeePC(params_D,'lqr','Hankel')   
    # start_deepc2=time.process_time()
    # xsim2, usim2, ysim2 = deepc2.loop(Tsim,A_list,B_list,C_list,D_list,x0)
    # end_deepc2=time.process_time()
    # print('Total Alberto running time: ', end_deepc2-start_deepc2)

    # deepc3 = DeePC(params_D,'lqr','Markov')   
    # x0 =  np.copy(xData[:, -1])
    # print('x0:',x0)
    # # print(x0)
    # start_deepc3=time.process_time()
    # xsim3, usim3, ysim3 = deepc3.loop(Tsim,A_list,B_list,C_list,D_list,x0)
    # end_deepc3=time.process_time()
    # print('Total DeepC running time: ', end_deepc3-start_deepc3)

    # print('x0',xData[:,-1])
    # x0 =  np.copy(xData[:, -1])
    # deepc3 = DeePC(params_D,'dis_lqr')   
    # start_deepc3=time.process_time()
    # xsim3, usim3, ysim3 = deepc3.loop(Tsim,A,B,C,D,x0)
    # end_deepc3=time.process_time()
    # print('Total Distributed Alberto running time: ', end_deepc3-start_deepc3)

    #     print(usim)

    ''' Plot results '''

    def plot_behavior(ax, title, xlabel, ylabel, data, label_prefix, ylim=None,log_scale=False):
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        for i in range(data.shape[0]):
            ax.plot(data[i,:], label=f'{label_prefix}{i}')

        if ylim is not None:
            ax.set_ylim(ylim)

        if log_scale:
            ax.set_yscale('log')

        ax.legend(loc='best')
        ax.grid(True)

    fig, ax = plt.subplots(4, figsize=(8, 10))

    # Plot control input
    # plot_behavior(ax[0], 'Control input', 'Time Steps', 'Input', usim, 'u', ylim=None)
    # # Plot output
    # plot_behavior(ax[1], 'State x1', 'Time Steps', 'x1', xsim[0,0,:].reshape(-1,Tsim+1), 'x', ylim=None)
    # # plot_behavior(ax[1], 'State x1', 'Time Steps', 'x1', xsim2[0,0,:].reshape(-1,Tsim), 'x', ylim=None)
    # plot_behavior(ax[2], 'State x2', 'Time Steps', 'x2', xsim[1,0,:].reshape(-1,Tsim+1), 'x', ylim=None)
    # # Plot output error
    # print('The reference:\n',wref[-p_central:])
    # error = np.abs(ysim2- np.tile(wref[-p_central:], Tsim))
    # print('eysim-wref.reshape(size_w,-1,order='F')[-2:,:]
    # plot_behavior(ax[3], 'Output error', 'Time Steps', 'Output error y - y_ref', error, 'y', ylim=None,log_scale=False)
    # plt.subplots_adjust(hspace=0.4)
    # plt.show()rror:\n',error[:,-1])
    # # error=

    # print('u_lqr',usim2[0,0])
    # print(usim_results3)
    mean_usim = usim_results.mean(axis=0)[0, :]  # Mean across experiments for the first control input
    std_usim = usim_results.std(axis=0)[0, :]    # Standard deviation for shading
    mean_usim2 = usim_results2.mean(axis=0)[0, :]  # Mean across experiments for the first control input
    std_usim2 = usim_results2.std(axis=0)[0, :]    # Standard deviation for shading
    # mean_usim3 = usim_results3.mean(axis=0)[0, :]  # Mean across experiments for the first control input
    # std_usim3 = usim_results3.std(axis=0)[0, :]    # Standard deviation for shading
    # mean_usim4 = usim_results4.mean(axis=0)[0, :]  # Mean across experiments for the first control input
    # std_usim4 = usim_results4.std(axis=0)[0, :] 

    mean_ysim = ysim_results.mean(axis=0)[0, :] 
    std_ysim = ysim_results.std(axis=0)[0, :]   
    mean_ysim2 = ysim_results2.mean(axis=0)[0, :]  
    std_ysim2 = ysim_results2.std(axis=0)[0, :]    
    # mean_ysim3 = ysim_results3.mean(axis=0)[0, :]  
    # std_ysim3 = ysim_results3.std(axis=0)[0, :]  
    # mean_ysim4 = ysim_results4.mean(axis=0)[0, :]  
    # std_ysim4 = ysim_results4.std(axis=0)[0, :] 

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim), mean_usim, label='Mean CVXPY', color='blue')
    plt.fill_between(np.linspace(0, Tsim,Tsim), mean_usim - std_usim, mean_usim + std_usim, color='blue', alpha=0.3)
    plt.plot(np.linspace(0, Tsim,Tsim), mean_usim2, label='Mean lqr', color='red')
    plt.fill_between(np.linspace(0, Tsim,Tsim), mean_usim2 - std_usim2, mean_usim2 + std_usim2, color='red', alpha=0.3)
    # plt.plot(np.linspace(0, Tsim,Tsim), mean_usim3, label='Mean Markov', color='black')
    # plt.fill_between(np.linspace(0, Tsim,Tsim), mean_usim3 - std_usim3, mean_usim3 + std_usim3, color='black', alpha=0.3)
    plt.ylabel('u')
    plt.title('Input')
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim), mean_ysim, label='Mean CVXPY', color='blue')
    plt.fill_between(np.linspace(0, Tsim,Tsim), mean_ysim - std_ysim, mean_ysim + std_ysim, color='blue', alpha=0.3)
    plt.plot(np.linspace(0, Tsim,Tsim), mean_ysim2, label='Mean lqr', color='red')
    plt.fill_between(np.linspace(0, Tsim,Tsim), mean_ysim2 - std_ysim2, mean_ysim2 + std_ysim2, color='red', alpha=0.3)
    # plt.plot(np.linspace(0, Tsim,Tsim), mean_ysim3, label='Mean Markov', color='black')
    # plt.fill_between(np.linspace(0, Tsim,Tsim), mean_ysim3 - std_ysim3, mean_ysim3 + std_ysim3, color='black', alpha=0.3)
    plt.ylabel('y')
    plt.title('Output')
    plt.grid(True)
    plt.legend()
    plt.show()


    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, Tsim,Tsim), usim[0,:], label='CVXPY', color='blue')
    # plt.plot(np.linspace(0, Tsim,Tsim), usim2[0,:], label='lqr', color='red')
    # # plt.plot(np.linspace(0, Tsim,Tsim), usim3[0,:], label='Markoc', color='black')
    # plt.ylabel('u')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, Tsim,Tsim), ysim[0,:], label='CVXPY', color='blue')
    # plt.plot(np.linspace(0, Tsim,Tsim), ysim2[0,:], label='lqr', color='red')
    # # plt.plot(np.linspace(0, Tsim,Tsim), ysim3[0,:], label='Markov', color='black')
    # plt.ylabel('y')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim[0,0,:], label='CVXPY', color='blue')
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim2[0,0,:], label='lqr', color='red')
    # # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim3[0,0,:], label='Markov', color='black')
    # plt.ylabel('phase')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim[1,0,:], label='CVXPY', color='blue')
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim2[1,0,:], label='lqr', color='red')
    # # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim3[1,0,:], label='Markov', color='black')
    # plt.ylabel('frequency')
    # plt.grid(True)
    # plt.legend()
    # plt.show()



    # Create a time array
    time = np.arange(xsim.shape[2])
    # data = {'time': time}
    # for method_num, x_array in enumerate([xsim, xsim2, xsim3], start=1):
    #     data[f'state_1_method_{method_num}'] = x_array[0, 0, :]  # Trajectory for state 1
    #     data[f'state_2_method_{method_num}'] = x_array[1, 0, :]  # Trajectory for state 2
    #     data[f'state_3_method_{method_num}'] = x_array[2, 0, :]  # Trajectory for state 3
    # # Create a DataFrame and save to CSV
    # df = pd.DataFrame(data)
    # df.to_csv('state_trajectories_ecc.csv', index=False)

    time2 = np.arange(usim.shape[1])
    data2 = {'time': time2}
    for method_num, u_array in enumerate([usim, usim2], start=1):
        data2[f'u_method_{method_num}'] = u_array[0, :]  # Trajectory for state 1
    for method_num, y_array in enumerate([ysim, ysim2], start=1):
        data2[f'y_method_{method_num}'] = y_array[0, :]  # Trajectory for state 1

    # Create a DataFrame and save to CSV
    df2 = pd.DataFrame(data2)
    df2.to_csv('w_ecc.csv', index=False)

        # def compute_metrics(t, response, ref, threshold=0.02):
        #     num_responses = response.shape[0]
        #     settling_times = []
        #     steady_state_values = response[:, -1]
            
        #     for i in range(num_responses):
        #         steady_state_value = response[i, -1]
        #         settling_threshold = np.abs(threshold * steady_state_value)
        #         # print('thre:',settling_threshold)
        #         settling_time_indices = np.where(np.abs(response[i, :] - steady_state_value) > settling_threshold)[0]
        #         # print(settling_time_indices)
        #         if settling_time_indices.size > 0:
        #             settling_time = t[settling_time_indices[-1]]
        #         else:
        #             settling_time = t[-1]
        #         settling_times.append(settling_time)
        #     steady_state_errors = np.abs(steady_state_values-ref )
        #     return np.max(settling_times), np.max(steady_state_errors)
        # settling_time, steady_state_error = compute_metrics(np.linspace(0, Tsim-1,Tsim), ysim, wref[-p_central:])

        # # print(f"Rise time: {rise_time:.2f} s")
        # print(f"Settling time: {settling_time:.2f} s")
        # # print(f"Overshoot: {overshoot:.2f} %")
        # print(f"Steady state error: {steady_state_error}")