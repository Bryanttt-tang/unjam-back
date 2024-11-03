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
    np.random.seed(1)
    v = 3 # number of units in the interconnected graph
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
        B_i = np.array([[0],
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

    A = np.array([[1, 0.1],
                   [-0.5,0.7]])
    B = np.array([[0],[1]])
    C = np.array([[0.25,0]])
    D = np.zeros(1)
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

 
    n = np.shape(A)[0]  # dimesion of the state
    m = np.shape(B)[1]  # dimesion of the input
    p = np.shape(C)[0]  # dimesion of the output
    q = p+m             # dimesion of input/output pair
    def create_connected_graph(n,topo='wheel'):
        """
        Creates a connected graph with n nodes and n-1 edges (a tree).
        # """
        # G = nx.Graph()
        # G.add_nodes_from(range(n))
        
        # # Create a minimum spanning tree to ensure connectivity
        # nodes = list(G.nodes())
        # np.random.shuffle(nodes)
        # for i in range(1, n):
        #     G.add_edge(nodes[i - 1], nodes[i])
        if topo=='chain':
            G = nx.path_graph(n)
        elif topo=='cycle':
            G = nx.cycle_graph(n)
        elif topo=='star':
            G = nx.star_graph(n - 1)
        elif topo=='wheel':
            G = nx.wheel_graph(n - 1)
        return G
    def create_reordered_mesh_graph(m, n):
        # Create a 2D grid graph (mesh grid)
        G_2d = nx.grid_2d_graph(m, n)
        
        # Map 2D coordinates (i, j) to 1D indices
        mapping = {(i, j): i * n + j for i in range(m) for j in range(n)}
        
        # Relabel the nodes to 1D ordering
        G_1d = nx.relabel_nodes(G_2d, mapping)
        
        return G_1d
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

    v_values = []
    markov_mean = []
    markov_var=[]
    lqr_mean = []
    lqr_var=[]
    dis_lqr_mean=[]
    dis_lqr_var=[]
    mean_sub=[]
    degree_values=[]
    max_sub=[]
    var_sub=[]
    mean_alter=[]
    mean_total=[]
    mean_lqr=[]
    mean_dis_lqr=[]
    mean_split=[]
    mean_split2=[]
    mean_inter=[]
    mean_proj2=[]
    mean_thread=[]
    dis_worst_mean=[]
    dis_worst_var=[]
    dis_theory_mean=[]
    dis_theory_var=[]
    cvx_mean=[]
    cvx_var=[]
    means = []
    maxs = []
    vars=[]
    cvx_time=[]
    for v in tqdm(np.arange(10, 110, 10)):
        graph = create_connected_graph(v,'chain') 

        n_subsystems = v
        # G = nx.path_graph(n_subsystems)
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
            neighbors = list(graph.neighbors(i))  # Get neighbors of subsystem i
            K_i = np.sum([k[i, j] for j in neighbors])  # Sum of k_ij for neighbors

            # Define the A_i matrix
            A_i = np.array([[1, d_t],
                            [-K_i / mi * d_t, 1 - (d[i] / mi) * d_t]])
            
            # Define the B_i matrix
            B_i = np.array([[0],
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

        A = np.array([[1, 0.1],
                    [-0.5,0.7]])
        B = np.array([[0],[1]])
        C = np.array([[0.25,0]])
        D = np.zeros(1)
        A_list = []
        B_list = []
        C_list = []
        for i in range(v):
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
        # graph=create_reordered_mesh_graph(v,v)
        # plt.figure(figsize=(8, 6))
        # pos = nx.spring_layout(graph)  # You can change the layout for different visualizations
        # nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
        # plt.title('Lattice Graph with Central Node')
        # plt.show()
        # v=v*v
        v_values.append(v)
        e=2*len(graph.edges()) # directed graph  
        v_con=nx.node_connectivity(graph)
        e_con=nx.edge_connectivity(graph)
        degree=max(dict(graph.degree()).values())
        degree_values.append(degree)
        print('Edges:',e)
        print('vertex connectivity:',v_con)
        print('edge connectivity:',e_con)
        print('degree:',degree)
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
        Tini = v*n+1 # length of the initial trajectory
        N = 5   # prediction horizon
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

        X0 = np.random.rand(n,v)  
        noise=0
        # X0=np.random.uniform(-100, 100, (n, v))#initial state of 10 units
        generator = generate_data(T,Tini,N,p,m,n,v,e,A_list,B_list,C_list,D_list,graph,0)
        xData, uData ,yData, uData_dis ,yData_dis,yData_noise= generator.generate_pastdata(X0)
        # print('max U:',np.max(uData))
        # print('max Y:',np.max(yData))
        # print('max X:',np.max(xData))
        wData=np.vstack((uData,yData))
        wData_dis=np.vstack((uData_dis,yData_dis))
        # print('uData:',uData[:,2])
        # print('yData:',yData[:,2])
        # print('uData_dis:',uData_dis[:,2])
        # print('wData:',wData[:,2])
        # print('wData_dis:',wData_dis[:,2])
        wini = wData[:, -Tini:].reshape(-1, 1, order='F')
        wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
        print(wini.shape) # (Tini*q=3*(m+p))
        print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

        # random_vector=np.zeros((q_central, N))
        # random_vector_dis=np.zeros((q_dis, N))
        random_vector=np.vstack((np.zeros((m_central,N)),0.25*np.ones((p_central, N)) ))
        random_vector_dis=np.vstack((np.zeros((m_dis,N)),0.25*np.ones((p_dis, N)) ))
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
        M = np.zeros(( len(pairs), q_dis))
        # Assign -1 and 1 to the corresponding places in M
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
        Up = H.Up
        Uf = H.Uf
        Yp = H.Yp
        Yf = H.Yf
        # h_total=H_dis.Hankel
        # print(h_total.shape)
        h=[]
        for i in range(len(H_j)):
            h.append(H_j[i].Hankel)
                
        max_iter=100
        dis_iter=5
        alpha=0.1
        # F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter)
        # lqr_exp_time=[]
        # dis_lqr_exp_time=[]
        
        lqr_exp_time=[]
        markov_exp_time=[]
        dis_lqr_exp_time=[]
        dis_lqr_worst_time=[]
        dis_lqr_theory_time=[]
        cvx_exp_time=[]
        for exp in range(1,6):
            F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter,[])
            wini = wData[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
            w_re=wini.reshape(-1,Tini,order='F')
            wini_dis = wData_dis[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')

            start_markovski = time.process_time()
            # Markovski solution of data-driven control
            W0=h_total[q_central*Tini:,:]
            # print('m*Tf+n:',m_central*N+n*v)
            # print('rank W0:',np.linalg.matrix_rank(W0))
            g_free=np.linalg.pinv(np.vstack((Up,Yp,Uf)))@np.vstack((wini, np.zeros((m_central*N,1)) ))
            y_free=Yf@g_free
            # y_free=F.matrix_vector_multiply(Yf,g_free)
            w_free=np.vstack((np.zeros((m_central,N)),y_free.reshape(-1,N))).reshape(-1, 1, order='F')
            # print('w_free \n',w_free)
            proj_markov=W0@np.linalg.pinv(W0.T@W0)@W0.T
            # start_markovski = time.process_time()    
            w_markov=proj_markov@(wref-w_free)+w_free
            end_markovski = time.process_time()
            markov_exp_time.append(end_markovski-start_markovski)

            start_alberto = time.process_time()
            w_split = F.lqr(wini, wref, Phi)
            # print('w_split \n',w_split[:q_central*Tini])
            end_alberto = time.process_time()
            print(f"Running time of Markovski Algo: ",end_markovski-start_markovski)
            print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto+F.lqr_off_time)
            lqr_exp_time.append(end_alberto-start_alberto)

            k=F.k_lqr
            print(F.k_lqr)
            start_dist = time.process_time()
            w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
            end_dist = time.process_time()
            print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist+F.dislqr_off_time)
            print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj)+F.dislqr_off_time)
            dis_lqr_exp_time.append(end_dist-start_dist)
            dis_lqr_worst_time.append(sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj)+F.dislqr_off_time)
            dis_lqr_theory_time.append(sum(F.worst_sub)+F.dislqr_off_time)
            print(len(F.worst_sub))
            print(f"Running time of theory case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_worst)+F.dislqr_off_time)

            # print(len(F.time_inter))
            # print(len(F.worst_sub))

            # print('inter:',sum(F.time_inter))
            # print('sub:',sum(F.worst_sub))
            # print('proj2:',sum(F.time_worst))
            # print('O(n):',sum(F.time_dis_lqr)-sum(F.time_alter_proj))

        #     start_cvx = time.process_time()
        #     g = cp.Variable((T-Tini-N+1,1))
        #     w_f = cp.Variable((N*q_central,1))
        #     objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))
        #     #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\
        #     #             + lambda_g*cp.norm(g, 1)          
        #     constraints = [ h_total[:Tini*q_central,:] @ g == wini,
        #                     h_total[Tini*q_central:,:] @ g == w_f,
        #                                 ]
        #     problem = cp.Problem(cp.Minimize(objective), constraints) 
        #     solver_opts = {
        #     'max_iter': 10000,
        #     'verbose': True     # Enable verbose output to debug
        # }
        #     # problem.solve(solver = cp.OSQP,**solver_opts)
        #     problem.solve(solver = cp.SCS)
        #     end_cvx = time.process_time()
        #     print('Running time of CVXPY for single LQR: ',end_cvx-start_cvx)
        #     cvx_exp_time.append(end_cvx-start_cvx)
        #     print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:]) )

        markov_mean.append(statistics.mean(markov_exp_time))
        markov_var.append(statistics.stdev(markov_exp_time))
        lqr_mean.append(statistics.mean(lqr_exp_time))
        lqr_var.append(statistics.stdev(lqr_exp_time))
        dis_lqr_mean.append(statistics.mean(dis_lqr_exp_time))
        dis_lqr_var.append(statistics.stdev(dis_lqr_exp_time))
        dis_worst_mean.append(statistics.mean(dis_lqr_worst_time))
        dis_worst_var.append(statistics.stdev(dis_lqr_worst_time))  
        # dis_theory_mean.append(statistics.mean(dis_lqr_theory_time))
        # dis_theory_var.append(statistics.stdev(dis_lqr_theory_time)) 
        # cvx_mean.append(statistics.mean(cvx_exp_time))
        # cvx_var.append(statistics.stdev(cvx_exp_time))

        # print('mean projection time of each subspace: ', statistics.mean(F.time_sub))
        # print('Max projection time of each subspace: ', max(F.time_sub))
        # print('Minimum projection time of each subspace: ', min(F.time_sub))
        # mean_sub.append(statistics.mean(F.time_sub))
        # var_sub.append(statistics.variance(F.time_sub))
        # max_sub.append(max(F.time_sub))
        # # min_sub.append(min(F.time_sub))
    
        
        # array_of_lists = np.array(F.all_sub)
        # mean_list = np.mean(array_of_lists, axis=0)
        # max_list = np.max(array_of_lists, axis=0)
        # variance_list = np.var(array_of_lists, axis=0)
        # mean_list = mean_list.tolist()
        # max_list = max_list.tolist()
        # variance_list = variance_list.tolist()
        # means.append(mean_list)
        # maxs.append(max_list)
        # vars.append(variance_list)
        

    # Create a dictionary to hold the results
    # results = {
    #     "means": means,
    #     "maxs": maxs,
    #     "varss": vars
    # }
    data = {'units':v_values, 'max_deg':degree_values,'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'distributed_lqr_mean':dis_lqr_mean,
            'distributed_lqr_var':dis_lqr_var, 'distributed_worst_mean':dis_worst_mean,'distributed_worst_var':dis_worst_var,
            'distributed_theory_mean':dis_theory_mean,'distributed_theory_var':dis_theory_var,'markovski_mean': markov_mean,'markovski_var': markov_var}
    # data = {'units':v_values, 'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'cvx_mean':cvx_mean,'cvx_var':cvx_var}
    df = pd.DataFrame(data)
    df.to_excel('results/10-100ecc,chain.xlsx', index=False)
    # data = {'units': v_values, 'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'distributed_lqr_mean':dis_lqr_mean,
    #         'distributed_lqr_var':dis_lqr_var, 'lqr_iteration':mean_lqr, 'dis_lqr_iteration':mean_dis_lqr, 'cvx':cvx_time,
    #         'average total proj':mean_total, 'mean thread':mean_thread,'average alternation proj':mean_alter, 'mean split':mean_split, 'mean split2':mean_split2, 
    #         'mean_inter': mean_inter, 'mean_proj2':mean_proj2,'mean_sub':mean_sub, 'var_sub':var_sub,'max_sub':max_sub}
    # df = pd.DataFrame(data)
    # df.to_excel('results/10-100,euler.xlsx', index=False)


    