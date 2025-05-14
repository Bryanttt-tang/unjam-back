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
if __name__ == "__main__":
    np.random.seed(123)
    # A = np.array([[1, 0.05],
    #                [-0.05,1]])
    # B = np.array([[0],[0.1]])
    # C = np.array([[0.1,0]])
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
    # Parameter study
    m_values = []
    lqr_time = []
    dis_lqr_time=[]
    mean_sub=[]
    max_sub=[]
    min_sub=[]
    mean_dis=[]
    mean_total=[]
    mean_split=[]
    mean_inter=[]
    mean_proj2=[]
    cvx_time=[]
    for m_input in tqdm(range(30,45,5)):
        m_values.append(m_input)
        B1 = 0.1/m_input*np.ones((2,m_input))
        
        controllability_matrix, is_controllable = check_controllability(A, B1)
        observability_matrix, is_observable = check_observability(A, C)
        print("Is the system controllable?", is_controllable)
        print("Is the system observable?",is_observable)
        ''' Integer invariants '''

        n = np.shape(A)[0]  # dimesion of the state
        m = np.shape(B1)[1]  # dimesion of the input # dimesion of the input
        p = np.shape(C)[0]  # dimesion of the output
        q = p+m             # dimesion of input/output pair
        graph='fully'
        v=50
        if graph=='chain':
            e=2*(v-1)
        elif graph=='fully':
            e=v*(v-1)
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
        T = v*m*L+v*n+100   # number of data points
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

        X0 = np.random.rand(n,v)         #initial state of 10 units
        generator = generate_data(T,Tini,N,p,m,n,v,e,A,B1,C,D,graph)
        xData, uData ,yData, uData_dis ,yData_dis = generator.generate_pastdata(X0)
        # print('Max x',np.max(xData))
        # print('Max Y',np.max(yData))
        wData=np.vstack((uData,yData))
        wData_dis=np.vstack((uData_dis,yData_dis))
        wini = wData[:, -Tini:].reshape(-1, 1, order='F')
        wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
        print('input:',m_input)
        print('wData:',wData.shape[0])
        print('wData_dis',wData_dis.shape[0])
        # wini = wData[:,10:10+Tini].reshape(-1,1, order='F')

        # random_vector_2=np.random.uniform(0, 5, size=(size_w, N))
        # r_u=np.zeros((m_deepc,1))
        # r_y=2*np.ones((p_deepc,1))
        # r=np.vstack((r_u,r_y))
        random_vector=np.zeros((q_central, N))
        random_vector_dis=np.zeros((q_dis, N))
        # wref=np.tile(r,N).reshape(-1,1, order='F')
        wref=random_vector.reshape(-1,1, order='F')
        wref_dis=random_vector_dis.reshape(-1,1, order='F')
        
        # Get M matrix
        # excluded_indices = []
        # excluded_indices.extend(list(range(0,m_input)))
        # for start in range(m_input+1, m_dis, (m_input+2)):
        #     excluded_indices.extend(range(start, start + m_input))
        
        # first_indices = [i for i in np.arange(m_dis) if i not in excluded_indices]
        # second_indices = []
        # for i in range(m_dis+1, q_dis):
        #     second_indices.append(i)
        #     second_indices.append(i - 1)
        # # Create pairs using broadcasting
        # pairs = np.transpose([first_indices, second_indices])
        # # print(pairs)
        # # Initialize the kernel matrix M
        # M = np.zeros(( len(pairs), q_dis))
        # # Assign -1 and 1 to the corresponding places in M
        # print('Crreating M matrix...')
        # for i, (idx1, idx2) in enumerate(pairs):
        #     M[i, idx1] = -1
        #     M[i, idx2] = 1
        # M_inv=np.linalg.inv(M@M.T)   
        
        excluded_indices=[]
        for start in range(0, m_dis, (m_input+v-1)):
            excluded_indices.extend(range(start, start + m_input))
        print(excluded_indices) 
        # excluded_indices =list(range(0, m_dis, v)) # fully graph
        print(m_dis)
        first_indices = [i for i in np.arange(m_dis) if i not in excluded_indices]
        sec = list(np.arange(m_dis, q_dis))
        second_indices = []

        # Iterate over range v
        for j in range(p_dis):
            # Append elements to second_indices, excluding the j-th element of sec
            second_indices.extend(k for idx, k in enumerate(sec) if idx != j)
        print(first_indices)
        print(second_indices)
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
        # pro=np.eye(q_deepc)-M.T@np.linalg.inv(M@M.T)@M
        # for i in range(q_deepc):
        #     print(pro[i])

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

        connected_components = find_connected_components(pairs)
        print('connected_components',connected_components)
        
        ''' Hankel matrix '''
        if graph =='chain':
            params_H = {"uData": uData,
                        "yData": yData,
                        "Tini": Tini,
                        "N": N,
                        "n":v*n,
                    }
            H = Hankel(params_H)
            
            # params_H_dis = {"uData": uData_dis,
            #             "yData": yData_dis,
            #             "Tini": Tini,
            #             "N": N,
            #             "n":v*n,
            #         }
            # H_dis = Hankel(params_H_dis)

            H_j=[]
            for j in range(v):
                if j==0:
                    params_H1 = {"uData": uData_dis[:(m+1),:],
                            "yData": yData_dis[:p,:],
                            "Tini": Tini,
                            "N": N,
                            "n":n,
                        }
                    H_j.append(Hankel(params_H1))
                elif j<v-1:
                    params_H1 = {"uData": uData_dis[m+1+(m+2)*(j-1):m+1+(m+2)*j,:],
                    "yData": yData_dis[j*p:(j+1)*p,:],
                    "Tini": Tini,
                    "N": N,
                    "n":n,
                    }
                    H_j.append(Hankel(params_H1))
                elif j==v-1:
                    params_H1 = {"uData": uData_dis[-(m+1):,:],
                    "yData": yData_dis[j*p:,:],
                    "Tini": Tini,
                    "N": N,
                    "n":n,
                    }
                    H_j.append(Hankel(params_H1))
                    
        elif graph=='fully':
            params_H = {"uData": uData,
                    "yData": yData,
                    "Tini": Tini,
                    "N": N,
                    "n":v*n,
                }
            H = Hankel(params_H)
            
            # params_H_dis = {"uData": uData_dis,
            #             "yData": yData_dis,
            #             "Tini": Tini,
            #             "N": N,
            #             "n":v*n,
            #         }
            # H_dis = Hankel(params_H_dis)

            H_j=[]
            for j in range(v):
                params_H1 = {"uData": uData_dis[(m+v-1)*j:(m+v-1)*(j+1),:],
                "yData": yData_dis[j*p:(j+1)*p,:],
                "Tini": Tini,
                "N": N,
                "n":n,
                }
                H_j.append(Hankel(params_H1))
                
            h_total=H.Hankel
            print(h_total.shape)
            h=[]
            print('creating hankel list...')
            for i in tqdm(range(len(H_j))):
                h.append(H_j[i].Hankel)
                
        max_iter=50
        dis_iter=2 
        alpha=0.1
        Tsim=100    
        F=functions(T,Tini, N, v, e, m_input,p, M, h_total, h, connected_components, alpha, max_iter, dis_iter)
        start_alberto = time.time()
        w_split = F.lqr(wini, wref,Phi,h_total)
        end_alberto = time.time()
        print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
        lqr_time.append(end_alberto-start_alberto)

        start_dist = time.time()
        w_split_dis = F.distributed_lqr(wini_dis, wref_dis,Phi_dis, h )
        end_dist = time.time()
        print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
        dis_lqr_time.append(end_dist-start_dist)
        
        print('mean projection time of each subspace: ', statistics.mean(F.time_sub))
        # print('Max projection time of each subspace: ', max(F.time_sub))
        # print('Minimum projection time of each subspace: ', min(F.time_sub))
        mean_sub.append(statistics.mean(F.time_sub))
        # max_sub.append(max(F.time_sub))
        # min_sub.append(min(F.time_sub))
        
        print(len(F.time_lqr))
        print('Projection time of total space: ',statistics.mean(F.time_lqr))
        mean_total.append(statistics.mean(F.time_lqr))
        # print(len(F.time_dis_lqr))
        mean_dis.append(statistics.mean(F.time_dis_lqr))
        print('Projection time of alternating projection: ',statistics.mean(F.time_dis_lqr))
        mean_split.append(statistics.mean(F.time_split))
        print('Spliting time: ',statistics.mean(F.time_split))
        mean_inter.append(statistics.mean(F.time_inter))
        print('Projection time of inter projection: ',statistics.mean(F.time_inter))
        mean_proj2.append(statistics.mean(F.time_proj2))
        print('Projection onto Cartisian of subspaces time: ',statistics.mean(F.time_proj2))
        
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
    #     cvx_time.append(end_cvx-start_cvx)
    # data = {'units': v_values, 'centralized_lqr': lqr_time, 'distributed_lqr':dis_lqr_time, 'average sub_proj':mean_sub,
                # 'min sub_proj':min_sub, 'max sub_proj':max_sub, 'average total proj':mean_total}
    data = {'num_inputs': m_values, 'centralized_lqr': lqr_time, 'distributed_lqr':dis_lqr_time,'average total proj':mean_total,
            'average alternation proj':mean_dis, 'mean split':mean_split, 'mean_inter': mean_inter, 'mean_proj2':mean_proj2, 'mean_sub':mean_sub}
    df = pd.DataFrame(data)
    df.to_excel('results/fully-parameter,30-45_50unit.xlsx', index=False)

    # print('The final cost of CVX:',problem.value)
    # print('The output trajectory using CVXPY: \n',w_f.value)
    # print('The output trajectory using DS-splitting: \n',w_split[size_w*Tini:])
    # print('The output trajectory using Distributed LQR: \n',w_split_dis[size_w*Tini:])
    # print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:]) )
    # print('The differnce between centralized and distributed methods: ',np.linalg.norm(w_split_dis[q_deepc*Tini:] - w_split[q_deepc*Tini:]) )

    