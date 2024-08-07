import numpy as np
import pandas as pd
from generator import generate_data, DeePC, Hankel,UnionFind
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json
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
    B = np.array([[0],[0.5]])
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
    # for m_input in range(5,35,5):
        # B1 = 1/m_input*np.ones((2,m_input))
        


    controllability_matrix, is_controllable = check_controllability(A, B)
    observability_matrix, is_observable = check_observability(A, C)
    print("Is the system controllable?", is_controllable)
    print("Is the system observable?",is_observable)
    ''' Integer invariants '''

    n = np.shape(A)[0]  # dimesion of the state
    m = np.shape(B)[1]  # dimesion of the input
    p = np.shape(C)[0]  # dimesion of the output
    q = p+m             # dimesion of input/output pair
    v_values = []
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
    graph='chain'
    for v in tqdm(np.arange(10, 55, 5)):
        v_values.append(v)
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

        X0 = np.random.rand(n,v)         #initial state of 10 units
        generator = generate_data(T,Tini,N,p,m,n,v,e,A,B,C,D,graph)
        xData, uData ,yData, uData_dis ,yData_dis = generator.generate_pastdata(X0)
        # print('Max x',np.max(xData))
        # print('Max Y',np.max(yData))
        wData=np.vstack((uData,yData))
        wData_dis=np.vstack((uData_dis,yData_dis))
        wini = wData[:, -Tini:].reshape(-1, 1, order='F')
        wini_dis = wData_dis[:, -Tini:].reshape(-1, 1, order='F')
        # print(wini.shape)
        # print(wini_dis.shape)
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
        if graph=='chain':
            excluded_indices =list(range(2, m_dis, 3))
            excluded_indices.append(0)
            first_indices = [i for i in np.arange(m_dis) if i not in excluded_indices]
            second_indices = []
            for i in range(m_dis+1, q_dis):
                second_indices.append(i)
                second_indices.append(i - 1) 
        elif graph=='fully':  
            excluded_indices =list(range(0, m_dis, (m+v-1))) # fully graph
            # print(excluded_indices)
            first_indices = [i for i in np.arange(m_dis) if i not in excluded_indices]
            sec = list(np.arange(m_dis, q_dis))
            second_indices = []
            for j in range(p_dis):
                # Append elements to second_indices, excluding the j-th element of sec
                second_indices.extend(k for idx, k in enumerate(sec) if idx != j)

        pairs = np.transpose([first_indices, second_indices])
        # print('pairs: \n',pairs)
        M = np.zeros(( len(pairs), q_dis))
        for i, (idx1, idx2) in enumerate(pairs):
            M[i, idx1] = -1
            M[i, idx2] = 1
        M_inv=np.linalg.inv(M@M.T)  

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
        # print('connected_components',connected_components)
        
        ''' Hankel matrix '''

        if graph =='chain':
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
            for j in range(v):
                if j==0:
                    params_H1 = {"uData": uData_dis[:2*m,:],
                            "yData": yData_dis[:p,:],
                            "Tini": Tini,
                            "N": N,
                            "n":n,
                        }
                    H_j.append(Hankel(params_H1))
                elif j<v-1:
                    params_H1 = {"uData": uData_dis[2+3*m*(j-1):2+3*m*j,:],
                    "yData": yData_dis[j*p:(j+1)*p,:],
                    "Tini": Tini,
                    "N": N,
                    "n":n,
                    }
                    H_j.append(Hankel(params_H1))
                elif j==v-1:
                    params_H1 = {"uData": uData_dis[2+3*m*(j-1):,:],
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
            
            params_H_dis = {"uData": uData_dis,
                        "yData": yData_dis,
                        "Tini": Tini,
                        "N": N,
                        "n":v*n,
                    }
            H_dis = Hankel(params_H_dis)

            H_j=[]
            for j in range(v):
                params_H1 = {"uData": uData_dis[v*m*j:v*m*(j+1),:],
                "yData": yData_dis[j*p:(j+1)*p,:],
                "Tini": Tini,
                "N": N,
                "n":n,
                }
                H_j.append(Hankel(params_H1))
        h_total=H.Hankel
        # print(h_total.shape)
        h=[]
        for i in range(len(H_j)):
            h.append(H_j[i].Hankel)       
        max_iter=50
        dis_iter=3 
        alpha=0.1
        Tsim=100    
        F=functions(T,Tini, N, v, e, m, p, M, h_total, h, connected_components, alpha, max_iter, dis_iter)
        lqr_exp_time=[]
        dis_lqr_exp_time=[]
        # for exp in range(1,6):
        #     wini = wData[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
        #     wini_dis = wData_dis[:, -(Tini+exp):-exp].reshape(-1, 1, order='F')
        #     start_alberto = time.process_time()
        #     w_split = F.lqr(wini, wref,Phi,h_total)
        #     end_alberto = time.process_time()
        #     print(f"Running time of Alberto Algo with {max_iter} iterations: ",end_alberto-start_alberto)
        #     lqr_exp_time.append(end_alberto-start_alberto)

        #     start_dist = time.process_time()
        #     w_split_dis = F.distributed_lqr(wini_dis, wref_dis,Phi_dis, h )
        #     end_dist = time.process_time()
        #     print(f"Running time of distributed Algo with {max_iter} outer iter and {dis_iter} alternating projection: ",end_dist-start_dist)
        #     dis_lqr_exp_time.append(end_dist-start_dist)
        # lqr_mean.append(statistics.mean(lqr_exp_time))
        # lqr_var.append(statistics.variance(lqr_exp_time))
        # dis_lqr_mean.append(statistics.mean(dis_lqr_exp_time))
        # dis_lqr_var.append(statistics.variance(dis_lqr_exp_time))
        # print('mean projection time of each subspace: ', statistics.mean(F.time_sub))
        # print('Max projection time of each subspace: ', max(F.time_sub))
        # print('Minimum projection time of each subspace: ', min(F.time_sub))
        # mean_sub.append(statistics.mean(F.time_sub))
        # var_sub.append(statistics.variance(F.time_sub))
        # max_sub.append(max(F.time_sub))
        # # min_sub.append(min(F.time_sub))
        
        # print(len(F.time_proj))
        # print('Projection time of total space: ',max(F.time_proj))
        # mean_total.append(statistics.mean(F.time_proj))
        # # print(len(F.time_dis_lqr))
        # mean_alter.append(statistics.mean(F.time_alter_proj))
        # print('Projection time of alternating projection: ',statistics.mean(F.time_alter_proj))
        # mean_lqr.append(statistics.mean(F.time_lqr))
        # mean_dis_lqr.append(statistics.mean(F.time_dis_lqr))
        # mean_thread.append(statistics.mean(F.time_thread))
        # print(len(F.time_thread))
        # mean_split.append(statistics.mean(F.time_split))
        # mean_split2.append(statistics.mean(F.time_split2))
        # mean_inter.append(statistics.mean(F.time_inter))
        # print('Projection time of inter projection: ',statistics.mean(F.time_inter))
        # mean_proj2.append(statistics.mean(F.time_proj2))
        # print('Projection onto Cartisian of subspaces time: ',statistics.mean(F.time_proj2))
        
        
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
        
        # CVXPY
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
    #     problem.solve(solver = cp.SCS,verbose=True)
    #     end_cvx = time.process_time()
    #     print('Running time of CVXPY for single LQR: ',end_cvx-start_cvx)
    #     cvx_time.append(end_cvx-start_cvx)
    # data = {'units': v_values, 'centralized_lqr': lqr_time, 'distributed_lqr':dis_lqr_time, 'average sub_proj':mean_sub,
                # 'min sub_proj':min_sub, 'max sub_proj':max_sub, 'average total proj':mean_total}
    # Create a dictionary to hold the results
    # results = {
    #     "means": means,
    #     "maxs": maxs,
    #     "varss": vars
    # }

    # # Save the results to a JSON file
    # with open('results.json', 'w') as f:
    #     json.dump(results, f)
    data = {'units': v_values, 'centralized_lqr_mean': lqr_mean, 'centralized_lqr_var': lqr_var, 'distributed_lqr_mean':dis_lqr_mean,
            'distributed_lqr_var':dis_lqr_var, 'lqr_iteration':mean_lqr, 'dis_lqr_iteration':mean_dis_lqr, 'cvx':cvx_time,
            'average total proj':mean_total, 'mean thread':mean_thread,'average alternation proj':mean_alter, 'mean split':mean_split, 'mean split2':mean_split2, 
            'mean_inter': mean_inter, 'mean_proj2':mean_proj2,'mean_sub':mean_sub, 'var_sub':var_sub,'max_sub':max_sub}
    df = pd.DataFrame(data)
    df.to_excel('results/10-50,euler.xlsx', index=False)

    # print('The final cost of CVX:',problem.value)
    # print('The output trajectory using CVXPY: \n',w_f.value)
    # print('The output trajectory using DS-splitting: \n',w_split[size_w*Tini:])
    # print('The output trajectory using Distributed LQR: \n',w_split_dis[size_w*Tini:])
    # print('The differnce between CVX and Alberto methods: ',np.linalg.norm(w_f.value - w_split[q_central*Tini:]) )
    # print('The differnce between centralized and distributed methods: ',np.linalg.norm(w_split_dis[q_deepc*Tini:] - w_split[q_deepc*Tini:]) )

    