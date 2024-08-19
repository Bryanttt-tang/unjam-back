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
import networkx as nx
if __name__ == "__main__":
    np.random.seed(123)
    # A = np.array([[1, 0.1],
    #                [-0.7,0.6]])
    # B = np.array([[0],[1]])
    # C = np.array([[0.25,0]])
    # D = np.zeros(1)
    A = np.array([[0.5, 0.05],
                [-0.05,0.5]])
    B = np.array([[0],[1]])
    C = np.array([[0.1,0]])
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
    def create_connected_graph(n):
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

    def add_random_edges2(G, num_edges=10):

        nodes = list(G.nodes())
        n = len(nodes)
        max_edges = n * (n - 1) // 2  # Maximum number of edges in a complete graph

        if G.number_of_edges() >= max_edges:
            print("Already a complete graph or cannot add more edges.")
            return

        existing_edges = set(G.edges())
        possible_edges = [(u, v) for u in nodes for v in nodes if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges]

        if len(possible_edges) < num_edges:
            print("Not enough possible edges to add.")
            return

        edges_to_add = np.random.choice(len(possible_edges), num_edges, replace=False)
        
        for idx in edges_to_add:
            u, v = possible_edges[idx]
            G.add_edge(u, v)

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
    e_values = []
    e_con_values=[]
    v_con_values=[]
    degree_values=[]
    lqr_mean = []
    lqr_var=[]
    dis_lqr_mean=[]
    dis_lqr_var=[]
    dis_worst_mean=[]
    dis_worst_var=[]
    mean_sub=[]
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
    means = []
    maxs = []
    vars=[]
    cvx_time=[]
    v = 100
    

    graph = create_connected_graph(v)   
    # Incrementally add edges until the graph becomes fully connected
    while graph.number_of_edges() <= 700:
        e=2*len(graph.edges()) # directed graph  
        v_con=nx.node_connectivity(graph)
        e_con=nx.edge_connectivity(graph)
        degree=max(dict(graph.degree()).values())
        e_values.append(e/2)
        e_con_values.append(e_con)
        v_con_values.append(v_con)
        degree_values.append(degree)
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
        Tini = 3# length of the initial trajectory
        N = 5   # prediction horizon
        L=Tini+N
        T = v*L+v*n+100   # number of data points
        R=1*np.eye(m_central)
        R_dis=1*np.eye(m_dis)
        # R[1,1]=0
        # R[3,3]=0
        Q=np.eye(p_dis)
        Phi=np.block([[R, np.zeros((R.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R.shape[1])), Q]])
        Phi_dis=np.block([[R_dis, np.zeros((R_dis.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R_dis.shape[1])), Q]])
        lambda_g = 0          # lambda parameter for L1 penalty norm on g

        # r = np.ones((p,1)) # reference trajectory, try to give different defitions of this to see if DeePC works!

        ''' Generate data '''

        # X0 = np.random.rand(n,v)  
        X0=np.random.uniform(-100, 100, (n, v))#initial state of 10 units
        generator = generate_data(T,Tini,N,p,m,n,v,e,A,B,C,D,graph)
        xData, uData ,yData, uData_dis ,yData_dis = generator.generate_pastdata(X0)
        # print('Max x',np.max(xData))
        # print('Max Y',np.max(yData))
        # print(' x',xData[:,2])
        # print(' Y',yData[:,0])
        wData=np.vstack((uData,yData))
        wData_dis=np.vstack((uData_dis,yData_dis))
        wini = wData[:, -Tini:].reshape(-1, 1, order='F')
        wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
        print(wini.shape) # (Tini*q=3*(m+p))
        print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

        random_vector=np.zeros((q_central, N))
        random_vector_dis=np.zeros((q_dis, N))
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
        # print(excluded_indices)
        # print(first_indices)
        # print(second_indices)
        # Create pairs using broadcasting
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
        # print(h_total.shape)
        h=[]
        for i in range(len(H_j)):
            h.append(H_j[i].Hankel)
                
        max_iter=100
        dis_iter=5 
        alpha=0.1
        Tsim=100    

        lqr_exp_time=[]
        dis_lqr_exp_time=[]
        dis_lqr_worst_time=[]
        
        for exp in range(1,4):
            F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter)
            wini = wData[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
            wini_dis = wData_dis[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
            start_alberto = time.process_time()
            w_split = F.lqr(wini, wref, Phi)
            end_alberto = time.process_time()
            print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
            lqr_exp_time.append(end_alberto-start_alberto)

            start_dist = time.process_time()
            w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
            end_dist = time.process_time()
            print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
            print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj))
            dis_lqr_exp_time.append(end_dist-start_dist)
            dis_lqr_worst_time.append(sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj))

        lqr_mean.append(statistics.mean(lqr_exp_time))
        lqr_var.append(statistics.variance(lqr_exp_time))
        dis_lqr_mean.append(statistics.mean(dis_lqr_exp_time))
        dis_lqr_var.append(statistics.variance(dis_lqr_exp_time))
        dis_worst_mean.append(statistics.mean(dis_lqr_worst_time))
        dis_worst_var.append(statistics.variance(dis_lqr_worst_time))
        print('mean projection time of each subspace: ', statistics.mean(F.time_sub))
        print('Max projection time of each subspace: ', max(F.time_sub))
        print('Minimum projection time of each subspace: ', min(F.time_sub))
        mean_sub.append(statistics.mean(F.time_sub))
        var_sub.append(statistics.variance(F.time_sub))
        max_sub.append(max(F.time_sub))
        # min_sub.append(min(F.time_sub))
        
        print(len(F.time_proj))
        print('Projection time of total space: ',max(F.time_proj))
        mean_total.append(statistics.mean(F.time_proj))
        # print(len(F.time_dis_lqr))
        mean_alter.append(statistics.mean(F.time_alter_proj))
        print('Projection time of alternating projection: ',statistics.mean(F.time_alter_proj))
        mean_lqr.append(statistics.mean(F.time_lqr))
        mean_dis_lqr.append(statistics.mean(F.time_dis_lqr))
        # mean_thread.append(statistics.mean(F.time_thread))
        # print(len(F.time_thread))
        mean_split.append(statistics.mean(F.time_split))
        mean_split2.append(statistics.mean(F.time_split2))
        mean_inter.append(statistics.mean(F.time_inter))
        print('Projection time of inter projection: ',statistics.mean(F.time_inter))
        # mean_proj2.append(statistics.mean(F.time_proj2))
        # print('Projection onto Cartisian of subspaces time: ',statistics.mean(F.time_proj2))
        if graph.number_of_edges() == v*(v-1)//2:
            break
        add_random_edges2(graph,50)
        # increase_edge_connectivity(graph, e_con+1)
        # if nx.edge_connectivity(graph) >10:
        #     break
        # add_random_edge(graph)
        assert nx.is_connected(graph), "The graph must remain connected"
        
    data = {'max_deg':degree_values,'edges': e_values, 'node_connectivity': v_con_values, 'edge_connectivity':e_con_values,'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 
            'distributed_lqr_mean':dis_lqr_mean,'distributed_lqr_var':dis_lqr_var, 'distributed_worst_mean':dis_worst_mean,'distributed_worst_var':dis_worst_var}
    df = pd.DataFrame(data)
    df.to_excel('results/degree-100units.xlsx', index=False)

        # start_alberto = time.time()
        # w_split = F.lqr(wini, wref, Phi, h_total)
        # end_alberto = time.time()
        # print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)

        # start_dist = time.time()
        # w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis, h)
        # end_dist = time.time()
        # print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
        # print('mean projection time of each subspace: ', statistics.mean(F.time_sub))
        # print('Max projection time of each subspace: ', max(F.time_sub))
        # print('Minimum projection time of each subspace: ', min(F.time_sub))
        # print(len(F.time_sub))
        # print(len(F.time_lqr))
        # print('Projection time of total space: ',statistics.mean(F.time_lqr))
        # print(len(F.time_dis_lqr))
        # print('Projection time of alternating projection: ',statistics.mean(F.time_dis_lqr))
        # print(statistics.mean(F.time_lqr))
        
        # CVXPY
    #     start_cvx = time.time()
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
    #     problem.solve(solver = cp.SCS,verbose=True)
    #     end_cvx = time.time()
    #     print('Running time of CVXPY for single LQR: ',end_cvx-start_cvx)
    #     # diff=np.linalg.norm(w_split-np.vstack((wini,wref)) )
    #     plt.plot(range(1, max_iter+1), np.squeeze(F.E))
    #     plt.plot(range(1, max_iter+1), np.squeeze(F.E_dis))
    #     plt.ylabel('Error')
    #     plt.legend(['Centralize', 'Distributed'])
    #     plt.title('Convergence Error of LQR')
    #     plt.grid(True)
    #     plt.show()
    #     # plt.show(block=False)
    #     # plt.pause(0.001)

    #     print('The final cost of CVX:',problem.value)
    #     # print('The output trajectory using CVXPY: \n',w_f.value)
    #     # print('The output trajectory using DS-splitting: \n',w_split[size_w*Tini:])
    #     # print('The output trajectory using Distributed LQR: \n',w_split_dis[size_w*Tini:])
    #     print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:]) )
        # print('The differnce between centralized and distributed methods: ',np.linalg.norm(w_split_dis[q_deepc*Tini:] - w_split[q_deepc*Tini:]) )
                
        
        
    # solver='lqr'
    # params_D = {'H': H, # an object of Hankel
    #             'H_dis':H_dis,
    #             'h_dis':h,
    #             'Phi':Phi,
    #             'Phi_dis':Phi_dis,
    #             'T':T,
    #             'Tini':Tini,
    #             'N':N,
    #             'n':n,
    #             'v':v,
    #             'e':e,
    #             'm':m,
    #             'p':p,
    #             'M':M,
    #             'connected_components':connected_components,
    #             'graph':graph,
    #             'alpha':alpha,
    #             'max_iter':max_iter,
    #             'dis_iter':dis_iter,
    #             'wref_dis' : wref_dis,
    #             'wref' : wref}

    # deepc = DeePC(params_D,solver)   
    # x0 =  xData[:, -1]
    # print(xData[:,-1])
    # # print(x0[:,9:10])
    # start_deepc=time.time()
    # usim, ysim = deepc.loop(Tsim,A,B,C,D,x0)
    # end_deepc=time.time()
    # print('Total DeepC running time: ', end_deepc-start_deepc)
    
    # #     print(usim)

    # ''' Plot results '''

    # def plot_behavior(ax, title, xlabel, ylabel, data, label_prefix, ylim=None):
    #     ax.set_title(title, fontsize=16, fontweight='bold')
    #     ax.set_xlabel(xlabel, fontsize=14)
    #     ax.set_ylabel(ylabel, fontsize=14)

    #     for i in range(data.shape[0]):
    #         ax.plot(data[i,:], label=f'{label_prefix}{i}')

    #     if ylim is not None:
    #         ax.set_ylim(ylim)

    #     ax.legend(loc='best')
    #     ax.grid(True)

    # fig, ax = plt.subplots(3, figsize=(8, 10))

    # # Plot control input
    # plot_behavior(ax[0], 'Control input', 'Time Steps', 'Input', usim, 'u', ylim=None)
    # # Plot output
    # plot_behavior(ax[1], 'Output Y', 'Time Steps', 'Output', ysim, 'y', ylim=None)
    # # Plot output error
    # print('The reference:\n',wref[-p_central:])
    # error = np.abs(ysim - np.tile(wref[-p_central:], Tsim))
    # print(error[:,-1])
    # # error=ysim-wref.reshape(size_w,-1,order='F')[-2:,:]
    # plot_behavior(ax[2], 'Output error', 'Time Steps', 'Output error y - y_ref', error, 'y', ylim=None)

    # # Adjust the space between plots
    # plt.subplots_adjust(hspace=0.4)

    # # Show the plot
    # plt.show()
    # plt.pause(0.001)


    