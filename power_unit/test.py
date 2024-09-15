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
    np.random.seed(123)
    A = np.array([[1, 0.1],
                   [-0.7,0.6]])
    B = np.array([[0],[1]])
    C = np.array([[0.025,0]])
    D = np.zeros(1)
    
    # A = np.array([[1, 0.05],
    #             [-0.05,0.5]])
    # B = np.array([[0],[0.1]])
    # C = np.array([[0.1,0]])
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
        Tini = 3 # length of the initial trajectory
        N = 5   # prediction horizon
        L=Tini+N
        T = v*L+v*n+100   # number of data points
        R=1*np.eye(m_central)
        R_dis=0.1*np.eye(m_dis)
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
        generator = generate_data(T,Tini,N,p,m,n,v,e,A,B,C,D,graph,noise)
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
        dis_lqr_exp_time=[]
        dis_lqr_worst_time=[]
        dis_lqr_theory_time=[]
        cvx_exp_time=[]
        for exp in range(1,6):
            F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter)
            wini = wData[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
            w_re=wini.reshape(-1,Tini,order='F')
            wini_dis = wData_dis[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
            start_alberto = time.process_time()
            w_split = F.lqr(wini, wref, Phi)
            # print('w_split \n',w_split[:q_central*Tini])
            end_alberto = time.process_time()
            print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
            lqr_exp_time.append(end_alberto-start_alberto)
            k=F.k_lqr
            print(F.k_lqr)
            start_dist = time.process_time()
            w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
            end_dist = time.process_time()
            print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
            print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj))
            dis_lqr_exp_time.append(end_dist-start_dist)
            dis_lqr_worst_time.append(sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj))
            dis_lqr_theory_time.append(sum(F.worst_sub))
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

        lqr_mean.append(statistics.mean(lqr_exp_time))
        lqr_var.append(statistics.stdev(lqr_exp_time))
        dis_lqr_mean.append(statistics.mean(dis_lqr_exp_time))
        dis_lqr_var.append(statistics.stdev(dis_lqr_exp_time))
        dis_worst_mean.append(statistics.mean(dis_lqr_worst_time))
        dis_worst_var.append(statistics.stdev(dis_lqr_worst_time))  
        dis_theory_mean.append(statistics.mean(dis_lqr_theory_time))
        dis_theory_var.append(statistics.stdev(dis_lqr_theory_time)) 
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
            'distributed_theory_mean':dis_theory_mean,'distributed_theory_var':dis_theory_var}
    # data = {'units':v_values, 'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'cvx_mean':cvx_mean,'cvx_var':cvx_var}
    df = pd.DataFrame(data)
    df.to_excel('results/10-100,dist-chain.xlsx', index=False)
    # data = {'units': v_values, 'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'distributed_lqr_mean':dis_lqr_mean,
    #         'distributed_lqr_var':dis_lqr_var, 'lqr_iteration':mean_lqr, 'dis_lqr_iteration':mean_dis_lqr, 'cvx':cvx_time,
    #         'average total proj':mean_total, 'mean thread':mean_thread,'average alternation proj':mean_alter, 'mean split':mean_split, 'mean split2':mean_split2, 
    #         'mean_inter': mean_inter, 'mean_proj2':mean_proj2,'mean_sub':mean_sub, 'var_sub':var_sub,'max_sub':max_sub}
    # df = pd.DataFrame(data)
    # df.to_excel('results/10-100,euler.xlsx', index=False)


    