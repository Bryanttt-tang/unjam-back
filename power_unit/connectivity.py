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
    A = np.array([[1, 0.1],
                   [-0.7,0.6]])
    B = np.array([[0],[1]])
    C = np.array([[0.25,0]])
    D = np.zeros(1)
    # A = np.array([[0.5, 0.05],
    #             [-0.05,0.5]])
    # B = np.array([[0],[1]])
    # C = np.array([[0.1,0]])
    # D = np.zeros(1)
    v = 10 # number of units in the interconnected graph
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
    EXP=2
    total_edge=30
    edge_incre=10
    lqr_exp_time=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    dis_lqr_exp_time=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    dis_lqr_worst_time=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    e_values = np.empty((EXP, (total_edge-v)//edge_incre+1) )
    e_con_values=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    v_con_values=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    degree_values=np.empty((EXP, (total_edge-v)//edge_incre+1) )
    for exp in range(EXP): 
        np.random.seed(exp)
        graph = create_connected_graph(v)  
        # Incrementally add edges until the graph becomes fully connected
        k=0
        while graph.number_of_edges() <= total_edge:
            e=2*len(graph.edges()) # directed graph  
            v_con=nx.node_connectivity(graph)
            e_con=nx.edge_connectivity(graph)
            degree=max(dict(graph.degree()).values())
            e_values[exp,k]=(e/2)
            e_con_values[exp,k]=(e_con)
            v_con_values[exp,k]=(v_con)
            degree_values[exp,k]=(degree)
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

            X0 = np.random.rand(n,v)  
            # X0=np.random.uniform(-, 3, (n, v))#initial state of 10 units
            generator = generate_data(T,Tini,N,p,m,n,v,e,A_list,B_list,C_list,D_list,graph,0)
            xData, uData ,yData, uData_dis ,yData_dis, yData_noise = generator.generate_pastdata(X0)
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

            
            # for exp in range(1,6):
            wini = wData[:, -Tini:].reshape(-1, 1, order='F')
            wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')

        #     g = cp.Variable((T-L+1,1))
        #     w_f = cp.Variable((N*q_central,1))
        #     objective = cp.quad_form(w_f-wref, psd_wrap(np.kron(np.eye(N),Phi)))
        #                 # + lambda_g*cp.norm(g, 1) 
        #     #             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\                  
        #     constraints = [ h_total[:Tini*q_central,:] @ g == wini,
        #                     h_total[Tini*q_central:,:] @ g == w_f
        #                                 ]
        #     # box constraint on inputs
        #     # w_f_reshaped = cp.reshape(w_f, (-1, N))
        #     # constraints += [
        #     #     w_f_reshaped[:m_central, :] <= 1,
        #     #     w_f_reshaped[:m_central, :] >= 1
        #     # ]
        #     problem = cp.Problem(cp.Minimize(objective), constraints) 
        #     solver_opts = {
        #     'max_iter': 10000,
        #     'verbose': True     # Enable verbose output to debug
        # }
        #     # problem.solve(solver = cp.OSQP,**solver_opts)
        #     problem.solve(solver = cp.SCS,verbose=False)

            F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter,0)
            start_alberto = time.process_time()
            w_split = F.lqr(wini, wref, Phi)
            end_alberto = time.process_time()
            print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
            lqr_exp_time[exp,k]=end_alberto-start_alberto

            start_dist = time.process_time()
            w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
            end_dist = time.process_time()
            print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
            print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj))
            dis_lqr_exp_time[exp,k]=end_dist-start_dist
            dis_lqr_worst_time[exp,k]=sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj)

            # lqr_mean.append(statistics.mean(lqr_exp_time))
            # lqr_var.append(statistics.variance(lqr_exp_time))
            # dis_lqr_mean.append(statistics.mean(dis_lqr_exp_time))
            # dis_lqr_var.append(statistics.variance(dis_lqr_exp_time))
            # dis_worst_mean.append(statistics.mean(dis_lqr_worst_time))
            # dis_worst_var.append(statistics.variance(dis_lqr_worst_time))

            if graph.number_of_edges() == v*(v-1)//2:
                break
            add_random_edges2(graph,edge_incre)
            k+=1
            # increase_edge_connectivity(graph, e_con+1)
            # if nx.edge_connectivity(graph) >10:
            #     break
            # add_random_edge(graph)
            assert nx.is_connected(graph), "The graph must remain connected"

    print('degree \n',degree_values)
    print(np.round(np.mean(degree_values,axis=0)))

    data = {'max_deg':np.round(np.mean(degree_values,axis=0)),'edges': np.round(np.mean(e_values,axis=0)), 'node_connectivity': np.round(np.mean(v_con_values,axis=0)), 'edge_connectivity':np.round(np.mean(e_con_values,axis=0)),'centralized_lqr_mean': np.mean(lqr_exp_time,axis=0), 'centralized_lqr_var': np.std(lqr_exp_time,axis=0)} 
            # 'distributed_lqr_mean':np.mean(dis_lqr_exp_time),'distributed_lqr_var':np.std(dis_lqr_exp_time), 'distributed_worst_mean':np.mean(dis_lqr_worst_time),'distributed_worst_var':np.std(dis_lqr_worst_time)}
    df = pd.DataFrame(data)
    df.to_excel('results/degree-100units.xlsx', index=False)