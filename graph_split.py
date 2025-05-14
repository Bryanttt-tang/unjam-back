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
    lqr_mean = []
    lqr_var=[]
    dis_lqr_mean=[]
    dis_lqr_var=[]
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
    # v = 100

    # graph = create_connected_graph(v) # create chain graph

    # Create a new graph
    graph = nx.Graph()

    # Mapping from 1D index to 2D grid coordinates
    index_to_coord = {}
    coord_to_index = {}

    # Create the first 3x3 grid network
    node_index = 0
    for i in range(3):
        for j in range(3):
            graph.add_node(node_index)
            index_to_coord[node_index] = (i, j)
            coord_to_index[(i, j)] = node_index
            node_index += 1

    # Connect nodes in the first grid
    for i in range(3):
        for j in range(3):
            current_index = coord_to_index[(i, j)]
            if i < 2:  # Vertical connections
                neighbor_index = coord_to_index[(i+1, j)]
                graph.add_edge(current_index, neighbor_index)
            if j < 2:  # Horizontal connections
                neighbor_index = coord_to_index[(i, j+1)]
                graph.add_edge(current_index, neighbor_index)

    # Offset for the second grid
    offset = 4

    # Create the second 3x3 grid network
    for i in range(3):
        for j in range(3):
            graph.add_node(node_index)
            index_to_coord[node_index] = (i+offset, j)
            coord_to_index[(i+offset, j)] = node_index
            node_index += 1

    # Connect nodes in the second grid
    for i in range(3):
        for j in range(3):
            current_index = coord_to_index[(i+offset, j)]
            if i < 2:  # Vertical connections
                neighbor_index = coord_to_index[(i+1+offset, j)]
                graph.add_edge(current_index, neighbor_index)
            if j < 2:  # Horizontal connections
                neighbor_index = coord_to_index[(i+offset, j+1)]
                graph.add_edge(current_index, neighbor_index)

    # Connect the two grids with three edges
    graph.add_edge(coord_to_index[(2, 2)], coord_to_index[(offset, 2)])
    graph.add_edge(coord_to_index[(2, 1)], coord_to_index[(offset, 1)])
    graph.add_edge(coord_to_index[(2, 0)], coord_to_index[(offset, 0)])

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=600, font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Two 3x3 Grid Networks Connected by Three Edges (1D Indexing)")
    plt.show()
    
    v=len(graph.nodes())
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
    N = 10   # prediction horizon
    L=Tini+N
    T = v*L+v*n+100   # number of data points
    R=0.1*np.eye(m_central)
    R_dis=0.1*np.eye(m_dis)
    # R[1,1]=0
    # R[3,3]=0
    Q=10*np.eye(p_dis)
    Phi=np.block([[R, np.zeros((R.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R.shape[1])), Q]])
    Phi_dis=np.block([[R_dis, np.zeros((R_dis.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R_dis.shape[1])), Q]])
    lambda_g = 1          # lambda parameter for L1 penalty norm on g
    lambda_1=1
    # r = np.ones((p,1)) # reference trajectory, try to give different defitions of this to see if DeePC works!

    ''' Generate data '''

    X0 = np.random.rand(n,v)  
    # X0=np.random.uniform(-100, 100, (n, v))#initial state of 10 units
    generator = generate_data(T,Tini,N,p,m,n,v,e,A,B,C,D,graph)
    xData, uData ,yData, uData_dis ,yData_dis = generator.generate_pastdata(X0)
    print('max U:',np.max(uData))
    print('max Y:',np.max(yData))
    print('max X:',np.max(xData))
    wData=np.vstack((uData,yData))
    wData_dis=np.vstack((uData_dis,yData_dis))

    wini = wData[:, -Tini:].reshape(-1, 1, order='F')
    wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
    print(wini.shape) # (Tini*q=3*(m+p))
    print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

    # random_vector=np.zeros((q_central, N))
    # random_vector_dis=np.zeros((q_dis, N))
    random_vector=np.vstack((np.zeros((m_central,N)),10*np.ones((p_central, N)) ))
    random_vector_dis=np.vstack((np.zeros((m_dis,N)),10*np.ones((p_dis, N)) ))
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

    # M-2
    graph2 = nx.path_graph(2)
    m_2=v*m+2*3
    q_2=m_2+p_central
    uData2=np.vstack((uData[:9,:],yData[9:12,:], uData[9:,:],yData[6:9,:]))
    first_2=[9,10,11,21,22,23]
    second_2=[33,34,35,30,31,32]
    pairs_2 = np.transpose([first_2, second_2])
    # print('pairs: \n',pairs)
    # Initialize the kernel matrix M
    M2 = np.zeros(( len(pairs_2), q_2))
    # Assign -1 and 1 to the corresponding places in M
    for i, (idx1, idx2) in enumerate(pairs_2):
        M2[i, idx1] = -1
        M2[i, idx2] = 1
    M_inv_2=np.linalg.inv(M2@M2.T)  
    print('Inter:\n',np.eye(M2.shape[1])-M2.T@M_inv_2@M2)
    connected_components_2 = find_connected_components(pairs_2)
    params_H_2_0 = {"uData": uData2[:12,:],
                "yData": yData[:9,:],
                "Tini": Tini,
                "N": N,
                "n":9*n,
            }
    params_H_2_1 = {"uData": uData2[12:,:],
                "yData": yData[9:,:],
                "Tini": Tini,
                "N": N,
                "n":9*n,
            }
    h2=[]
    h2.append(Hankel(params_H_2_0).Hankel)
    h2.append(Hankel(params_H_2_1).Hankel)
    wData2=np.vstack((uData2,yData_dis))
    wini_2 = wData2[:, -Tini:].reshape(-1, 1, order='F')
    print(wini_2.shape) # (Tini*q_dis=3*(m^2+p))

    random_vector_2=np.vstack((np.zeros((m_2,N)),10*np.ones((p_central, N)) ))
    wref_2=random_vector_2.reshape(-1,1, order='F')
    R_2=0.1*np.eye(m_2)
    Phi_2=np.block([[R_2, np.zeros((R_2.shape[0],Q.shape[1]))], [np.zeros((Q.shape[0],R_2.shape[1])), Q]])
    # # M-3
    # m_3=v*m+2*9
    # q_3=m_3+p_central
    # uData3=np.vstack((uData[:9,:],  , uData[9:,:]))

    # # M-4
    # m_4=v*m+2*15
    # q_4=m_4+p_central
    # uData4=np.vstack((uData[:9,:],  , uData[9:,:]))


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
    Tsim=200    
    F=functions(T,Tini, N, v, e, m, 1, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter)
    F2=functions(T,Tini, N, 2, 2, 9, 3, 9, M2, h_total, h2, connected_components_2, graph2, alpha, max_iter, dis_iter)
    # F3=functions(T,Tini, N, v, e, m, p, M3, h_total, h3, connected_components, graph3, alpha, max_iter, dis_iter)
    # F4=functions(T,Tini, N, v, e, m, p, M4, h_total, h4, connected_components, graph4, alpha, max_iter, dis_iter)
    # lqr_exp_time=[]
    # dis_lqr_exp_time=[]
    
# case 1
    wini = wData[:, -Tini:].reshape(-1, 1, order='F')
    wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
    start_alberto = time.process_time()
    w_split = F.lqr(wini, wref, Phi)
    end_alberto = time.process_time()
    print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
#case 2
    # start_case2 = time.process_time()
    # w_split_dis = F2.distributed_lqr(wini_2, wref_2, Phi_2)
    # end_case2 = time.process_time()
    # print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_case2-start_case2)
    # print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F2.time_inter)+sum(F2.time_worst))
# case 4
    start_dist = time.process_time()
    w_split_dis = F.distributed_lqr(wini_dis, wref_dis, Phi_dis)
    end_dist = time.process_time()
    print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
    
    print(f"Running time of Worst case distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",sum(F.time_inter)+sum(F.time_worst)+sum(F.time_dis_lqr)-sum(F.time_alter_proj) )
    print(max(F.time_sub))
    # print(sum(F.time_dis_lqr)-sum(F.time_alter_proj))
    # print(F.k_lqr)