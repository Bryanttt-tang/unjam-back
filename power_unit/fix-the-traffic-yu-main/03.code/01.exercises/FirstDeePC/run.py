""" Script to run the simulation """
import numpy as np
from scipy.linalg import expm
from generator import generate_data, DeePC, Hankel
import matplotlib.pyplot as plt
# from sippy import system_identification

""" The objective of this script should be to verify the correct construction of the Hankel matrix and to understand when
    DeePC works and when it breaks down, a hint on how to explore this is to produce Hankel matrices with full rank (good case)
    and with lower ranks (bad case) and see how DeePC behaves in each case. 

    Also explore what happens when changing the costs Q,R, the parameter lambda_g and the prediction horizon N 
    or the length of the initial trajectory Tini. 

    Please produce some plots to illustrate your findings.
"""

if __name__ == "__main__":
    
    # ''' Problem definition '''
    A = np.array([[1.01, 0.01, 0.00],
                   [0.01,1.01,0.01],
                   [0.00,0.01,1.01]])
    B = np.eye(3)
    C = np.eye(3)
    D = np.zeros((3,3))
    
    # A = np.array([[1, 0.1],
    #                [-0.7,0.6]])
    # B = np.array([[0],[1]])
    # C = np.array([[0.25,0]])
    # D = np.zeros(1)
    # B_aug=np.kron(np.eye(2),B)
    # C_aug=np.kron(np.eye(2),C)
    # A_aug=np.kron(np.eye(2),A)
    # # A_aug[0,2]=0.25
    # A_aug[1,2]=0.25
    # # A_aug[2,0]=0.25
    # A_aug[3,0]=0.25

    # # print('A_aug:\n',A_aug)
    # # print('B_aug:\n',B_aug)
    # A=A_aug
    # B=B_aug
    # C=C_aug
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

    ''' Simulation parameters '''
    Tsim =100 # number of simulation points
    Tini = 3 # length of the initial trajectory
    N = 5   # prediction horizon
    L=Tini+N
    T = L+n+100

    Q= 1*np.eye(p)        # cost matrix on current output y
    R= 1*np.eye(m)        # cost matrix on the control imput applied
    lambda_g = 0          # lambda parameter for L1 penalty norm on g

    ''' Generate data '''
    
    X0 = np.random.rand(n,1)         #initial state
    generator = generate_data(T,Tini,N,p,m,n,A,B,C,D)
    xData, uData, yData = generator.generate_pastdata(X0)

    from scipy.io import savemat

    savemat('data.mat', {'uData': uData, 'yData': yData})


    # identified_model = system_identification(yData, uData, method='N4SID', SS_order=3, SS_fixed_order=True)
    # print(identified_model)
    r = 1*np.ones((p,1)) # reference trajectory, try to give different defitions of this to see if DeePC works!
    print('r',r)
    print('max U:',np.max(uData))
    print('max Y:',np.max(yData))
    print('max X:',np.max(xData))

    ''' Hankel matrix '''

    params_H = {"uData": uData,
                "yData": yData,
                "Tini": Tini,
                "N": N,
                "n":n,
               }
    
    H = Hankel(params_H)

    U, S, VT = np.linalg.svd(H.Hankel)

    # Truncate to rank 27
    rank = np.linalg.matrix_rank(H.Hankel)
    U_truncated = U[:, :rank]  # Shape (48, 27)
    check=U_truncated.T@U_truncated-np.eye(rank)
    S_truncated = np.diag(S[:rank])  # Shape (27, 27)

    # Form the truncated matrix representing the subspace
    H_truncated = U_truncated @ S_truncated  # Resulting shape (48, 27)

    # H_truncated now represents your reduced matrix in the lower-dimensional subspace
    print("Truncated Hankel matrix shape:", H_truncated.shape)
    # H.construct_Hankel()

    ''' Simulation'''
    wref=np.tile( np.vstack(( np.zeros((m,1)),r)),N).reshape(-1, 1, order='F')
    
    params_D = {'H': H,
            'Q': Q,
            'R': R,
            'lambda_g' : lambda_g,
            'r' : r,
            'wref': wref}
    solver='CVXPY'
    deepc = DeePC(params_D,solver)   
    x0 =  xData[:, -1].reshape(-1, 1, order='F')
    print('x0:',x0)
    usim, ysim = deepc.loop(Tsim,A,B,C,D,x0)
    # print(usim)

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

    fig, ax = plt.subplots(2, figsize=(10, 15))

    # Plot control input
    plot_behavior(ax[0], 'Control input', 'Time Steps', 'Input', usim, 'u', ylim=None)
    print('input:',usim[:,-1])
    # print('input ref',-A.dot(r))
    # Plot output error
    error = np.abs(ysim - np.tile(r, Tsim))
    print(ysim[:,-1])
    print('error:\n',error[:,-1])
    plot_behavior(ax[1], 'Output error', 'Time Steps', 'Output error y - y_ref', error, 'y', ylim=None)

    # Adjust the space between plots
    plt.subplots_adjust(hspace=0.4)

    # Show the plot
    plt.show()
    

    def compute_metrics(t, response, ref,threshold=0.02):
        num_responses = response.shape[0]
        settling_times = []
        steady_state_values = response[:, -1]
        
        for i in range(num_responses):
            steady_state_value = response[i, -1]
            settling_threshold = np.abs(threshold * 1)
            # print('thre:',settling_threshold)
            settling_time_indices = np.where(np.abs(response[i, :] - steady_state_value) > settling_threshold)[0]
            # print(settling_time_indices)
            if settling_time_indices.size > 0:
                settling_time = t[settling_time_indices[-1]]
            else:
                settling_time = t[-1]
            settling_times.append(settling_time)
        steady_state_errors = np.abs(steady_state_values-ref )
        return np.max(settling_times), np.max(steady_state_errors)
    # print(ysim.shape)

    # Compute performance metrics
    settling_time, steady_state_error = compute_metrics(np.linspace(0, Tsim,Tsim), ysim,r)

    # print(f"Rise time: {rise_time:.2f} s")
    print(f"Settling time: {settling_time:.2f} s")
    # print(f"Overshoot: {overshoot:.2f} %")
    print(f"Steady state error: {steady_state_error}")
