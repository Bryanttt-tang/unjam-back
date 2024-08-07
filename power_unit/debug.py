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
    A = np.array([[-1, 0, 2],
                [-2, -3, -4],
                [1,0,-1]])
    B = np.array([[1,1],[0,2],[-1,3]])
    C = np.array([[1,0,0],[0,1,0]])
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

    v = 1 # number of units in the interconnected graph
    

    graph = create_connected_graph(v)   
    # increase_edge_connectivity(graph, 9)
    # Incrementally add edges until the graph becomes fully connected
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
    lambda_g = 1          # lambda parameter for L1 penalty norm on g
    lambda_1=1
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
    print('uData:',uData[:,2])
    print('yData:',yData[:,2])
    print('uData_dis:',uData_dis[:,2])
    print('wData:',wData[:,2])
    print('wData_dis:',wData_dis[:,2])
    wini = wData[:, -Tini:].reshape(-1, 1, order='F')
    wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
    print(wini.shape) # (Tini*q=3*(m+p))
    print(wini_dis.shape) # (Tini*q_dis=3*(m^2+p))

    # random_vector=np.zeros((q_central, N))
    # random_vector_dis=np.zeros((q_dis, N))
    random_vector=np.vstack((np.zeros((m_central,N)),np.zeros((p_central, N)) ))
    random_vector_dis=np.vstack((np.zeros((m_dis,N)),np.zeros((p_dis, N)) ))
    # wref=np.tile(r,N).reshape(-1,1, order='F')
    wref=random_vector.reshape(-1,1, order='F')
    wref_dis=random_vector_dis.reshape(-1,1, order='F')
    
    
    
    ''' Hankel matrix '''
    
    params_H = {"uData": uData,
                "yData": yData,
                "Tini": Tini,
                "N": N,
                "n":v*n,
            }
    H = Hankel(params_H)
    

    max_iter=50
    dis_iter=10 
    alpha=0.1
    Tsim=100    
    # lqr_exp_time=[]
    # dis_lqr_exp_time=[]
    


    
    Tsim=1000  
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

    deepc = DeePC(params_D,solver)   
    x0 =  xData[:, -1]
    print('x0:',xData[:,-1])
    # print(x0[:,9:10])
    start_deepc=time.time()
    usim, ysim = deepc.loop(Tsim,A,B,C,D,x0)
    end_deepc=time.time()
    print('Total DeepC running time: ', end_deepc-start_deepc)

    #     print(usim)

    ''' Plot results '''

    def plot_behavior(ax, title, xlabel, ylabel, data, label_prefix, ylim=None):
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        for i in range(data.shape[0]):
            ax.plot(data[i,:], label=f'{label_prefix}{i}')

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend(loc='best')
        ax.grid(True)

    fig, ax = plt.subplots(3, figsize=(8, 10))

    # Plot control input
    plot_behavior(ax[0], 'Control input', 'Time Steps', 'Input', usim, 'u', ylim=None)
    # Plot output
    plot_behavior(ax[1], 'Output Y', 'Time Steps', 'Output', ysim, 'y', ylim=None)
    # Plot output error
    print('The reference:\n',wref[-p_central:])
    error = np.abs(ysim - np.tile(wref[-p_central:], Tsim))
    print(error[:,-1])
    # error=ysim-wref.reshape(size_w,-1,order='F')[-2:,:]
    plot_behavior(ax[2], 'Output error', 'Time Steps', 'Output error y - y_ref', error, 'y', ylim=None)

    # Adjust the space between plots
    plt.subplots_adjust(hspace=0.4)

    # Show the plot
    plt.show()
        # plt.pause(0.001)


    