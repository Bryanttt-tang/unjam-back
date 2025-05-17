import numpy as np
import cvxpy as cp
import time
from tqdm import tqdm
from cvxpy.atoms.affine.wraps import psd_wrap
from functions import functions
"""Generic class to define past data, initial trajectory """
# np.random.seed(123)
class UnionFind:
        def __init__(self):
            self.parent = {}
            self.rank = {}

        def find(self, node):
            if self.parent[node] != node:
                self.parent[node] = self.find(self.parent[node])  # path compression
            return self.parent[node]

        def union(self, node1, node2):
            root1 = self.find(node1)
            root2 = self.find(node2)

            if root1 != root2:
                # union by rank
                if self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                elif self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                else:
                    self.parent[root2] = root1
                    self.rank[root1] += 1

        def add(self, node):
            if node not in self.parent:
                self.parent[node] = node
                self.rank[node] = 0

class generate_data():
    
    def __init__(self,T,Tini,N,p,m,n,v,e,A,B,C,D,graph,noise):
        
        
        self.T = T
        self.Tini = Tini  # shift window
        self.N = N # time horizon
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.p = p
        self.n = n
        self.m = m
        self.v= v
        self.e= e
        self.graph=graph
        self.noise=noise
        
    def generate_pastdata(self,x0):
        
        T = self.T
        p = self.p # output dimension
        n= self.n  # state dimension
        m = self.m # input dimension
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        v= self.v
        e= self.e
    
        # augmented manifest variable
        uData_dis = np.empty(( (v*m+e*p),T)) 
        yData_dis = np.empty((v*p,T))
        # original manifest variable
        uData = np.empty(( v*m,T)) 
        yData = np.empty((v*p,T))
        yData_noise = np.empty((v*p,T))

        
        xData = np.empty((v*n,T+1))
        x=x0 # X0 = np.random.rand(n,10)  
        # y=C[0]@x
        # print('v',v)
        for t in tqdm(range(self.T)):
            # u = np.array(np.random.randn(m, v))
            u=np.random.uniform(-10, 10, (m, v))
            all_u = []
            y=np.zeros((p,v))
            # print('y.shape',y.shape)
            for i in range(v):
                # print(C[i].shape)
                y[:,i:i+1]=C[i]@x[:,i:i+1]
            for i in range(v):
                # print(x[:,i:i+1].shape)
                # print(y[:,i+1].shape)
                # y[:,i:i+1]=C[i]@x[:,i:i+1]
                x[:,i:i+1] = A[i]@x[:,i:i+1]+B[i]@(u[:,i:i+1])
                all_u.append(u[:, i:i+1])
                sum_term = np.zeros_like(B[i] @ y[:, i:i+1])
                # print(i)
                # print(list(self.graph.neighbors(i)))
                neighbors = sorted(list(self.graph.neighbors(i)))  # Sort neighbors

                for j in neighbors:
                    sum_term += B[i] @ ( y[:, j:j+1])
                    all_u.append(y[:,j:j+1])
                x[:,i:i+1]+= sum_term
            
            # print(np.vstack(all_u).shape)
            # print('noise',np.random.normal(0, 0.1, yData[:,[t]].shape))
            uData_dis[:,[t]] = np.vstack(all_u)
            yData_dis[:,[t]] = y.reshape(-1, 1,order='F')
            uData[:,[t]] = u.reshape(-1, 1,order='F')
            yData[:,[t]] = y.reshape(-1, 1,order='F')
            yData_noise[:,[t]] = y.reshape(-1, 1,order='F') + np.random.normal(1, np.sqrt(self.noise), yData[:,[t]].shape)
            xData[:,[t+1]] = x.reshape(-1, 1, order='F')
         
        # u_mean = uData.mean(axis=1, keepdims=True)
        # y_mean = yData.mean(axis=1, keepdims=True)
        # x_mean = xData.mean(axis=1, keepdims=True)
        # u_std = uData.std(axis=1, keepdims=True)
        # y_std = yData.std(axis=1, keepdims=True)
        # x_std = xData.std(axis=1, keepdims=True)
        # uData = (uData - u_mean) / u_std
        # yData = (yData - y_mean) / y_std
        # xData = (xData - x_mean) / x_std

        return  xData, uData ,yData, uData_dis ,yData_dis, yData_noise
    

"""the minimal representation ( i.e. the smallest state dimension) in this case is 3 as X(k) stays in R^3
Note that the number of data points T must be at least (m + 1)(t + n(B)) − 1 in order to satisfy the persistency
of excitation condition. In this case n=m=p=3"""

"""The Hankel matrix used in DeePC is not Square!!! it has T-Tini-N +1 columns and has (Tini +N)*m rows 
    T >= (m+1)(Tini+N+n(B)-1)"""

"""The reference trajectory is in R^Np"""

"""Hankel Matrix class"""

class Hankel():
    
    def __init__(self,params):
        
        self.uData = params['uData']
        self.yData = params['yData']
        self.wData = np.vstack((self.uData, self.yData))
        self.T = self.uData.shape[1] # the number of offline data
        self.Tini = params['Tini']  # past data
        self.N = params['N']  # time horizon/ future data
        self.L = self.Tini + self.N  # The order L of the Hankel matrix
        self.m = self.uData.shape[0]
        self.p = self.yData.shape[0]
        self.n = params['n']
        
        """ Initialize the empty Hankel matrix"""
        # COMPLETE THIS FUNCTION
        self.Hankel=np.empty((self.L*(self.m+self.p) , self.T-self.L+1)) 
        self.Hankel_u=np.empty((self.L*self.m , self.T-self.L+1)) 
        self.Hankel_y=np.empty((self.L*self.p , self.T-self.L+1)) 
        self.construct_Hankel()
        
        [Up, Uf] = np.vsplit(self.Hankel_u, [self.Tini * self.m])
        [Yp, Yf] = np.vsplit(self.Hankel_y, [self.Tini * self.p])
        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        
        self.check_rank()   # Check if fundamental lemma holds

        
    def construct_Hankel(self): # hankel matrix is H(w) where w=col(u,y)
        # COMPLETE THIS FUNCTION
        """The Hankel matrix used in DeePC is not Square!!! it has T-Tini-N +1 columns and has (Tini +N)*m rows 
        T >= (m+1)(Tini+N+n(B)-1)"""
        
        for i in range(self.L):
                self.Hankel[i*(self.m+self.p):(i+1)*(self.m+self.p), :]=self.wData[:, i:i+(self.T+1-self.L)]
                self.Hankel_u[i*self.m:(i+1)*self.m, :]=self.uData[:, i:i+(self.T+1-self.L)]
                self.Hankel_y[i*self.p:(i+1)*self.p, :]=self.yData[:, i:i+(self.T+1-self.L)]
        # pass
      
         
    def check_rank(self):
        
        rank = np.linalg.matrix_rank(self.Hankel)
        print("Shape of Hankel", self.Hankel.shape)
        # print("Uf:", self.Uf)
        if rank == self.L*self.m + self.n:  #  L=Tini + N 
            
            print("The Hankel matrix spans the restricted behaviour: ",
                  "rank(H) = Lm+n = ", rank)
        else: 
            print("The Hankel matrix does NOT span the restricted behavior:",
                  "rank(H) =", rank, "\n",
                  "mL+n =", self.L*self.m +self.n)
            
            
class DeePC():
    
    def __init__(self,params_D,solver,represent,exp):
        
        self.H = params_D['H'] # Hankel matrix object
        self.H_dis = params_D['H_dis'] # Hankel matrix object
        self.h_dis = params_D['h_dis'] # a list containing hankel matrix of each unit
        self.Q = params_D['Q']
        self.R = params_D['R']
        self.Phi=params_D['Phi']
        self.Phi_dis=params_D['Phi_dis']
        self.T = params_D['T']
        self.Tini = params_D['Tini']
        self.N = params_D['N']
        self.ML=params_D['ML'] # Markovsky matrix
        self.L=self.Tini+self.N
        self.n = params_D['n']
        self.v = params_D['v']
        self.e = params_D['e']
        self.exp=exp
        self.xData=params_D['xData']
        self.m=params_D['m'] # num_input of each unit
        self.m_total= self.m*self.v # total manifest variable
        self.m_dis = self.m_total+self.e
        self.p=params_D['p'] # num_output of each unit
        self.p_total=self.p*self.v
        self.q_total=self.m_total+self.p_total
        self.q_dis=self.m_dis+self.p_total
        self.M=params_D['M']
        self.connected_components = params_D['connected_components']
        self.graph=params_D['graph']
        self.alpha = params_D['alpha']
        self.max_iter = params_D['max_iter']
        self.dis_iter = params_D['dis_iter']
#         self.lambda_g = params_D['lambda_g'] # lambda parameter for L1 penalty norm on g
        self.wref = params_D['wref'] # the reference trajectory to be tracked
        self.wref_dis = params_D['wref_dis']
        self.w_star = params_D['w_star']
        self.solver=solver
        self.represent=represent
        self.E=[]
        self.A_aug=params_D['A_aug']
        self.B_aug=params_D['B_aug']
        self.C_aug=params_D['C_aug']
        if self.represent=='Hankel':
            self.F=functions(self.T, self.Tini,self.N, self.v,self.e, self.m, 1, self.p, self.M, self.H.Hankel, self.h_dis, self.connected_components, self.graph, self.alpha,self.max_iter,self.dis_iter,self.w_star)
        elif self.represent=='Markov':
            self.F=functions(self.T, self.Tini,self.N, self.v,self.e, self.m, 1, self.p, self.M, self.ML, self.h_dis, self.connected_components, self.graph, self.alpha,self.max_iter,self.dis_iter,self.w_star)
    
    def solve_deepc(self,uini,yini):
        # COMPLETE THIS FUNCTION
        u_reshape=uini.reshape(self.m_total,-1,order='F')
        y_reshape=yini.reshape(self.p_total,-1,order='F')
#         print('uini:\n',u_reshape)
#         print('yini:\n',y_reshape)
        wini=np.vstack((u_reshape,y_reshape)).reshape(-1,1, order='F')
        
        g=cp.Variable((self.T-self.L+1, 1))
        # g=cp.Variable((self.F.rank_total, 1))
        w_f=cp.Variable((self.q_total*self.N , 1)) 
        
        objective = cp.quad_form(w_f-self.wref, psd_wrap(np.kron(np.eye(self.N),self.Phi)))
#             + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\
#             + lambda_g*cp.norm(g, 1)          

        constraints = [ self.F.h_total[:self.Tini*self.q_total,:] @ g == wini,
                            self.F.h_total[self.Tini*self.q_total:,:] @ g == w_f,
                            ]
            # box constraint on inputs
        w_f_reshaped = cp.reshape(w_f, (-1, self.N))
        constraints += [
            w_f_reshaped[:self.m_total, :] <= 0.5,
            w_f_reshaped[:self.m_total, :] >= -0.5,
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints) 
        problem.solve(solver='SCS', warm_start=True)
        e=problem.value
#         print('Current g:',g.value)
        # print('g:',g.value)
        # print('Constraints:',self.H.Uf @ g)
        # print('u:',u.value)
        return (g.value.reshape(-1,1, order='F'), w_f.value.reshape(-1,1, order='F'),e)
    
    def solve_mpc(self,xini):
        # COMPLETE THIS FUNCTION

        x0=xini
        # print('x0.shape',x0.shape)
        x = cp.Variable((self.v*self.n, self.N+1))  # State trajectory
        u = cp.Variable((self.m_total, self.N))    # Control input trajectory
        y= cp.Variable((self.p_total, self.N))
        # Objective function and constraints
        cost = 0
        constraints = [x[:, 0:1] == x0.reshape(-1,1)]  # Initial state constraint

        for k in range(self.N):
            # Quadratic cost function for tracking
            cost += cp.quad_form(y[:, k:k+1] - self.wref[self.m_total:self.m_total+self.p_total], self.Q) + cp.quad_form(u[:, k:k+1]-self.wref[:self.m_total], self.R)
            
            # System dynamics constraint
            constraints += [y[:, k] == self.C_aug @ x[:, k]]
            constraints += [x[:, k+1] == self.A_aug @ x[:, k] + self.B_aug @ u[:, k]]

        # Solve the optimization problem
        problem_mpc = cp.Problem(cp.Minimize(cost), constraints)
        problem_mpc.solve()

        return (np.vstack((u.value, y.value)).reshape(-1, 1, order='F'))

    def solve_lqr(self,uini,yini):
        u_reshape=uini.reshape(self.m_total,-1,order='F')
        y_reshape=yini.reshape(self.p_total,-1,order='F')

        wini=np.vstack((u_reshape,y_reshape)).reshape(-1,1, order='F')
        w = self.F.lqr(wini, self.wref, self.Phi)
        return w[self.q_total*self.Tini:]
    
    def solve_dist_lqr(self,uini,yini):
        u_reshape=uini.reshape(self.m_dis,-1,order='F')
        y_reshape=yini.reshape(self.p_total,-1,order='F')
#         print('uini:\n',u_reshape)
#         print('yini:\n',y_reshape)
        wini=np.vstack((u_reshape,y_reshape)).reshape(-1,1, order='F')
        w = self.F.distributed_lqr(wini, self.wref_dis, self.Phi_dis)
        return w[self.q_dis*self.Tini:]
    
    def get_next_input(self, uini, yini,xini,s = 1):
        # Return the first control action in the predicted optimal sequence
        if self.solver=='CVXPY':
            (g, w_f,e) = self.solve_deepc(uini, yini) # u is u*
            self.E.append(e)
            w_f=w_f.reshape(self.q_total,-1,order='F')
            u=w_f[:self.m_total,:].reshape(-1,1, order='F')
            return u[:self.m_total*s].reshape(-1,1, order='F')  # apply (u(t),...,u(t+s))=(u_0*,...,u_s*)
        elif self.solver=='lqr':
            w_f=self.solve_lqr(uini, yini)
            w_f=w_f.reshape(self.q_total,-1,order='F')
            u=w_f[:self.m_total,:].reshape(-1,1, order='F')
            return u[:self.m_total*s].reshape(-1,1, order='F')
        elif self.solver=='dis_lqr':
            w_f=self.solve_dist_lqr(uini, yini)
            w_f=w_f.reshape(self.q_dis,-1,order='F')
            u=w_f[:self.m_dis,:].reshape(-1,1, order='F')
            return u[:self.m_dis*s].reshape(-1,1, order='F')
        elif self.solver=='MPC':
            w_f=self.solve_mpc(xini)
            w_f=w_f.reshape(self.q_total,-1,order='F')
            u=w_f[:self.m_total,:].reshape(-1,1, order='F')
            return u[:self.m_total*s].reshape(-1,1, order='F')
        elif self.solver=='NoControl':
            u=np.zeros((self.m_total*self.N,1))
            return u[:self.m_total*s].reshape(-1,1, order='F')
        
    def get_wini(self):
        if self.solver=='dis_lqr':
            uini = self.H_dis.uData[:, -self.Tini:].reshape(-1, 1, order='F')
            yini = self.H_dis.yData[:, -self.Tini:].reshape(-1, 1, order='F') 
        else:    
            uini = self.H.uData[:, -(self.Tini+self.exp):-self.exp].reshape(-1, 1, order='F')
            yini = self.H.yData[:, -(self.Tini+self.exp):-self.exp].reshape(-1, 1, order='F')
            # uini = self.H.uData[:, -self.Tini:].reshape(-1, 1, order='F')
            # yini = self.H.yData[:, -self.Tini:].reshape(-1, 1, order='F')
        
        return uini ,yini
    
    def loop(self,Tsim,A,B,C,D,x0):
        
        if self.solver=='dis_lqr':
            usim = np.empty((self.m_total, Tsim)) # in total, Tsim simulation points
            ysim = np.empty((self.p_total, Tsim))
            xsim = np.empty((self.n, self.v, Tsim+1))
            uini,yini = self.get_wini()
            xini=self.xData[:,-1]
            x=np.copy(x0).reshape(self.n, -1, order='F') # X0 = np.random.rand(n,10)  
            xsim[:,:,0]=np.copy(x)
            for t in tqdm(range(Tsim)):
                
                u = self.get_next_input(uini, yini,xini)
                # print('u_shape',u.shape)
                y=C@x
    #             print(y.shape)
                base_index=0
                for i in range(self.v):
                    neighbors = sorted(self.graph.neighbors(i)) # to distinguish externel and inter inputs
                    # print('u',u[base_index].reshape(-1, 1))
                    # print('base',base_index)
                    x[:,i:i+1] = A[i]@x[:,i:i+1]+B[i]@(u[base_index].reshape(-1, 1))
                    usim[i, [t]] = u[base_index]
                    base_index += len(neighbors)+1
                    
                    sum_term = np.zeros_like(B @ y[:, i:i+1])
                    # print('sun_term',sum_term.shape)
                    for j in neighbors:
                        sum_term += B @ (y[:, j:j+1])
                    x[:,i:i+1]+= sum_term
                
                # usim[:, [t]] = u
                # print(usim[:, [t]])
                ysim[:, [t]] = y.reshape(-1, 1)
                xsim[:,:,t+1]=np.copy(x)
                uini = np.vstack((uini[self.m_dis:], u))
                # uini is extended with the contents of u vertically (along the rows), with the first self.H.m rows of uini being excluded. 
                yini = np.vstack((yini[self.p_total:], y.reshape(-1, 1)))
                xini=np.copy(x).reshape(-1, 1,order='F')

        else:
            usim = np.empty((self.m_total, Tsim))
            ysim = np.empty((self.p_total, Tsim))
            xsim = np.empty((self.n, self.v, Tsim+1))
            uini,yini = self.get_wini()
            # xini=self.xData[:,-self.exp-1]
            xini=self.xData[:,-1]
            x=np.copy(x0).reshape(self.n, -1, order='F') # X0 = np.random.rand(n,10)  
            xsim[:,:,0]=np.copy(x)
            for t in tqdm(range(Tsim)):
                u = self.get_next_input(uini, yini,xini)
                
                y=C[0]@x
                # print(y)
                # for i in range(self.v):
                #     y[:,i:i+1]=C[i]@x[:,i:i+1]
                for i in range(self.v):
                    # x[:,i:i+1] = A@x[:,i:i+1]+B@(u[i].reshape(-1, 1)+np.sum(y,axis=1, keepdims=True) - self.v*self.p*y[:, i])
                    # print('x:',x[:,i:i+1].shape)
                    # print('u',u[i].reshape(-1, 1))
                    y[:,i:i+1]=C[i]@x[:,i:i+1]
                    x[:,i:i+1] = A[i]@x[:,i:i+1]+B[i]@(u[i].reshape(-1, 1))
                    sum_term = np.zeros_like(B[i] @ y[:, i:i+1])
                    # print(sum_term.shape)
                    neighbors = sorted(list(self.graph.neighbors(i)))  # Sort neighbors
                    for j in neighbors:
                        sum_term += B[i] @ (y[:, j:j+1])
                    x[:,i:i+1]+= sum_term
                    if t==0:
                        x[1,i:i+1]+=2
                        u[i]+=2
                    if t==50:
                        x[1,i:i+1]+=5
                        u[i]+=5
                
                usim[:, [t]] = u
                if t==0:
                    usim[:, [t]]-=2
                if t==50:
                    usim[:, [t]]-=5
                ysim[:, [t]] = y.reshape(-1, 1)
                xsim[:,:,t+1]=np.copy(x)
                uini = np.vstack((uini[self.m_total:], u))
                # uini is extended with the contents of u vertically (along the rows), with the first self.H.m rows of uini being excluded. 
                yini = np.vstack((yini[self.p_total:], y.reshape(-1, 1)))
                xini=np.copy(x).reshape(-1, 1,order='F')
        return xsim, usim, ysim