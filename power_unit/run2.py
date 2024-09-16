import numpy as np
import pandas as pd
from generator import generate_data, DeePC, Hankel,UnionFind
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import statistics
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from functions import functions
from multiprocessing.pool import ThreadPool
import networkx as nx
if __name__ == "__main__":
    # np.random.seed(1)
    A = np.array([[1, 0.1],
                   [-0.5,0.7]])
    B = np.array([[0],[1]])
    C = np.array([[0.25,0]])
    D = np.zeros(1)

    A = np.array([[1.01, 0.01],
                   [0.00,0.01]])
    B = np.eye(2)
    C = np.eye(2)
    # D = np.zeros((3,3))
    # A = np.array([[-1, 0, 2],
    #             [-2, -3, -4],
    #             [1,0,-1]])
    # B = np.array([[1,1],[0,2],[-1,3]])
    # C = np.array([[1,0,0],[0,1,0]])
    D = np.zeros(1)
    eigenvalues, _ = np.linalg.eig(A)
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
    # m = np.shape(B)[1]  # dimesion of the input
    # p = np.shape(C)[0]  # dimesion of the output
    m=1
    p=1
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

    v = 2 # number of units in the interconnected graph

    graph = create_connected_graph(v)   
    # increase_edge_connectivity(graph, 2)

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
    ''' Simulation parameters '''
    Tini = 3 # length of the initial trajectory
    N = 5  # prediction horizon
    L=Tini+N
    T = v*L+v*n+100   # number of data points
    R=1*np.eye(m_central)
    R_dis=1*np.eye(m_dis)
    # R[1,1]=0
    # R[3,3]=0
    Q=1*np.eye(p_dis)
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
    generator = generate_data(T,Tini,N,p,m,n,v,e,A,B,C,D,graph,0)
    xData, uData ,yData, uData_dis ,yData_dis, yData_noise = generator.generate_pastdata(X0)
    print(np.var(uData))

    # print('var in original',np.var(yData))
    # print('max U:',np.max(uData))
    # print('max Y:',np.max(yData))
    # print('max X:',np.max(xData))
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
    print(wini.shape) # (Tini*q=3*(m+p))
    print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

    # random_vector=np.zeros((q_central, N))
    # random_vector_dis=np.zeros((q_dis, N))
    random_vector=np.vstack((np.zeros((m_central,N)),0*np.ones((p_central, N)) ))
    random_vector_dis=np.vstack((np.zeros((m_dis,N)),0*np.ones((p_dis, N)) ))
    # wref=np.tile(r,N).reshape(-1,1, order='F')
    wref=random_vector.reshape(-1,1, order='F')
    wref_dis=random_vector_dis.reshape(-1,1, order='F')
    
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
    # Assign -1 and 1 to the corresponding places in M
    M = np.zeros(( len(pairs), q_dis))
    for i, (idx1, idx2) in enumerate(pairs):
        M[i, idx1] = -1
        M[i, idx2] = 1
    M_inv=np.linalg.inv(M@M.T)  
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

    params_H_noise = {"uData": uData,
                "yData": yData_noise,
                "Tini": Tini,
                "N": N,
                "n":v*n,
            }
    H_noise = Hankel(params_H_noise)
    
    params_H_dis = {"uData": uData_dis,
                "yData": yData_dis,
                "Tini": Tini,
                "N": N,
                "n":v*n,
            }
    H_dis = Hankel(params_H_dis)

    H_j=[]
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
        start_row = end_row
        
    h_total=H.Hankel
    # h_total=H_dis.Hankel
    # print(h_total.shape)
    h=[]
    for i in range(len(H_j)):
        h.append(H_j[i].Hankel)
            
    max_iter=2000
    dis_iter=1
    alpha=0.1
    num_runs=10
    cost_data = np.zeros((num_runs, max_iter))
    F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter)
    # F_noise=functions(T,Tini, N, v, e, m, 1, p, M, H_noise.Hankel, h, connected_components, graph, alpha, max_iter, dis_iter)
    # lqr_exp_time=[]
    # dis_lqr_exp_time=[]        
    wini = wData[:, -Tini:].reshape(-1, 1, order='F')
    wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
    start_alberto = time.process_time()
    w_split = F.lqr(wini, wref, Phi)
    end_alberto = time.process_time()
    print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
    plt.figure(figsize=(10, 6))
    print('wini',wini)
    print('w_split',w_split)
    print('final error', F.E1[-1])
    # plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E))
    # plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E0))
    plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E1))
    # plt.plot(range(0, max_iter+1), np.squeeze(F_noise.E))
    # # plt.plot(range(0, max_iter+1), np.squeeze(F.E_dis))
    plt.ylabel('Error')
    # plt.legend([ 'wini','wf'])
    # plt.title('Convergence Error of LQR')
    plt.grid(True)
    plt.show()

    # w_split_noise = F_noise.lqr(wini, wref, Phi)
    # start_dist = time.process_time()
    # w_split_dis = F.distributed_lqr(wini_dis, wref_dis,Phi_dis)
    # end_dist = time.process_time()
    # print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
    # # print(f"Running time of worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)

    # random_vector_dis=np.random.uniform(0, 10, size=(q_dis, L))
    # w_ran_dis=random_vector_dis.reshape(-1,1, order='F')
    # errors = []
    # direct_projection=F.proj(h_total,w_ran_dis)
    # # direct_projection=h_total@np.linalg.pinv(h_total)@w_ran
    # with ThreadPool(processes=8) as pool:
    #     for ite in tqdm(range(1,26)):
    #         projected_point_alternating = F.alternating_projections(F.proj_h_sub, M,M_inv, w_ran_dis, pool, num_iterations=ite)
    #         error = np.linalg.norm(direct_projection-projected_point_alternating)
    #         errors.append(error)
    # # plt.cla()
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 26), errors, marker='o')
    # plt.xlabel('Iteration')
    # plt.ylabel('Error')
    # plt.title('Difference between direct Projection vs. Alternating Projection (System with 2 units)')
    # plt.grid(True)
    # plt.show()   

    U, S, VT = np.linalg.svd(h_total)
    rank_total = np.linalg.matrix_rank(h_total)
    U_truncated = U[:, :rank_total]
    # CVXPY
    start_cvx = time.time()
    g = cp.Variable((T-L+1,1))
    w_f = cp.Variable((N*q_central,1))
    objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))+ lambda_g*cp.norm(g, 1) 
    #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\
                    
    constraints = [ h_total[:Tini*q_central,:] @ g == wini,
                    h_total[Tini*q_central:,:] @ g == w_f
                                ]
    # box constraint on inputs
    w_f_reshaped = cp.reshape(w_f, (-1, N))
    constraints += [
        w_f_reshaped[:m_central, :] <= 1,
        w_f_reshaped[:m_central, :] >= 1
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints) 
    solver_opts = {
    'max_iter': 10000,
    'verbose': True     # Enable verbose output to debug
}
    # problem.solve(solver = cp.OSQP,**solver_opts)
    problem.solve(solver = cp.OSQP,verbose=False)
    end_cvx = time.time()
    print('Running time of CVXPY for single LQR: ',end_cvx-start_cvx)
    # diff=np.linalg.norm(w_split-np.vstack((wini,wref)) )
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E))
    # plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E0))
    # plt.plot(range(0, F.k_lqr[0]+1), np.squeeze(F.E1))
    # plt.plot(range(0, max_iter+1), np.squeeze(F_noise.E))
    # # plt.plot(range(0, max_iter+1), np.squeeze(F.E_dis))
    # plt.ylabel('Error')
    # plt.legend([ 'wini','wf'])
    # plt.title('Convergence Error of LQR')
    plt.grid(True)
    plt.show()
    # plt.show(block=False)
    # plt.pause(0.001)
    # g_off=np.linalg.inv(U_truncated[:Tini*q_central,:])@wini
    print('iterations:',F.k_lqr)
    # w_f_off=U_truncated[Tini*q_central:,:]@np.linalg.pinv(U_truncated[:Tini*q_central,:])@wini
    w_f_off=h_total[Tini*q_central:,:]@np.linalg.pinv(h_total[:Tini*q_central,:])@wini
    print('The final cost of CVX:',problem.value)
    print('cvx\n',w_f.value)
    print('alberto\n',w_split[q_central*Tini:])
#     # print('The output trajectory using CVXPY: \n',w_f.value)
#     # print('The output trajectory using DS-splitting: \n',w_split[size_w*Tini:])
#     # print('The output trajectory using Distributed LQR: \n',w_split_dis[size_w*Tini:])
    print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:]) )
    print('The differnce between offline CVX and Alberto methods: ',np.linalg.norm(w_f_off - w_split[q_central*Tini:]) )
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

    Tsim=100  
    solver='CVXPY'
    params_D = {'H': H, # an object of Hankel
                'H_dis':H_dis,
                'h_dis':h,
                'Phi':Phi,
                'Phi_dis':Phi_dis,
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
                'wref' : wref}

    # deepc = DeePC(params_D,'CVXPY')   
    # x0 =  np.copy(xData[:, -1])
    # print('x0:',xData[:,-1])
    # print(x0)
    # start_deepc=time.process_time()
    # xsim, usim, ysim = deepc.loop(Tsim,A,B,C,D,x0)
    # end_deepc=time.process_time()
    # print('Total DeepC running time: ', end_deepc-start_deepc)
    # print('x0',xData[:,-1])

    x0 =  np.copy(xData[:, -1])
    deepc2 = DeePC(params_D,'lqr')   
    start_deepc2=time.process_time()
    xsim2, usim2, ysim2 = deepc2.loop(Tsim,A,B,C,D,x0)
    end_deepc2=time.process_time()
    print('Total Alberto running time: ', end_deepc2-start_deepc2)
    # print('wini',wini)
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
    plot_behavior(ax[0], 'Control input', 'Time Steps', 'Input', usim2, 'u', ylim=None)
    # Plot output
    plot_behavior(ax[1], 'State x1', 'Time Steps', 'x1', xsim2[0,0,:].reshape(-1,Tsim+1), 'x', ylim=None)
    # plot_behavior(ax[1], 'State x1', 'Time Steps', 'x1', xsim2[0,0,:].reshape(-1,Tsim), 'x', ylim=None)
    plot_behavior(ax[2], 'State x2', 'Time Steps', 'x2', xsim2[1,0,:].reshape(-1,Tsim+1), 'x', ylim=None)
    # Plot output error
    print('The reference:\n',wref[-p_central:])
    error = np.abs(ysim2- np.tile(wref[-p_central:], Tsim))
    print('error:\n',error[:,-1])
    # error=ysim-wref.reshape(size_w,-1,order='F')[-2:,:]
    plot_behavior(ax[3], 'Output error', 'Time Steps', 'Output error y - y_ref', error, 'y', ylim=None,log_scale=False)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim), usim[0,:], label='CVXPY', color='blue')
    plt.plot(np.linspace(0, Tsim,Tsim), usim2[0,:], label='lqr', color='red')
    # plt.plot(np.linspace(0, Tsim,Tsim), usim3[0,:], label='dis_lqr', color='black')
    plt.ylabel('u')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim), ysim[0,:], label='CVXPY', color='blue')
    plt.plot(np.linspace(0, Tsim,Tsim), ysim2[0,:], label='lqr', color='red')
    # plt.plot(np.linspace(0, Tsim,Tsim), ysim3[0,:], label='dis_lqr', color='black')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim+1), xsim[0,0,:], label='CVXPY', color='blue')
    plt.plot(np.linspace(0, Tsim,Tsim+1), xsim2[0,0,:], label='lqr', color='red')
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim3[0,0,:], label='dis_lqr', color='black')
    plt.ylabel('phase')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, Tsim,Tsim+1), xsim[1,0,:], label='CVXPY', color='blue')
    plt.plot(np.linspace(0, Tsim,Tsim+1), xsim2[1,0,:], label='lqr', color='red')
    # plt.plot(np.linspace(0, Tsim,Tsim+1), xsim3[1,0,:], label='dis_lqr', color='black')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create a time array
    time = np.arange(xsim.shape[2])
    data = {'time': time}
    for method_num, x_array in enumerate([xsim, xsim2], start=1):
        data[f'state_1_method_{method_num}'] = x_array[0, 0, :]  # Trajectory for state 1
        data[f'state_2_method_{method_num}'] = x_array[1, 0, :]  # Trajectory for state 2

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('state_trajectories_box.csv', index=False)

    time2 = np.arange(usim.shape[1])
    data2 = {'time': time2}
    for method_num, u_array in enumerate([usim, usim2], start=1):
        data2[f'u_method_{method_num}'] = u_array[0, :]  # Trajectory for state 1
    for method_num, y_array in enumerate([ysim, ysim2], start=1):
        data2[f'y_method_{method_num}'] = y_array[0, :]  # Trajectory for state 1

    # Create a DataFrame and save to CSV
    df2 = pd.DataFrame(data2)
    df2.to_csv('w_box.csv', index=False)

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