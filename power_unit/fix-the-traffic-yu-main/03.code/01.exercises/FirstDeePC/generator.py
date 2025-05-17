import numpy as np
import cvxpy as cp
from tqdm import tqdm

"""Generic class to define past data, initial trajectory """

class generate_data():
    
    def __init__(self,T,Tini,N,p,m,n,A,B,C,D):
        
        
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
        
    def generate_pastdata(self,x0):
        
        T = self.T
        p = self.p # output dimension
        n= self.n  # state dimension
        m = self.m # input dimension
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        
        yData = np.empty((p,T))          # offline Output data Matrix y^d
        uData = np.empty((m,T))          # offline Input data matrix u^d
        xData = np.empty((n,T+1))        # State data Matrix         
        x = x0
        u= np.ones((m,1))

        for t in range(self.T):
            u = np.array(np.random.randn(m, 1)) # random input (to ensure the persistency of excitation)
            # u=np.random.uniform(-10, 10, (m, 1))
            x = A@x+B@u
            y = C@x
            
            uData[:,[t]] = u
            yData[:,[t]] = y
            xData[:,[t+1]] = x

        # u_mean = np.mean(uData,axis=1,keepdims=True)
        # y_mean = np.mean(yData,axis=1,keepdims=True)
        # x_mean = np.mean(xData,axis=1,keepdims=True)
        # u_std = np.std(uData,axis=1,keepdims=True)
        # y_std = np.std(yData,axis=1,keepdims=True)
        # print('y_mean',y_mean)
        # print('y_std',y_std)
        # x_std = np.std(xData,axis=1,keepdims=True)
        # # uData = (uData - u_mean) / u_std
        # yData = (yData - y_mean) / y_std
        # # xData = (xData - x_mean) / x_std

        # u_max = np.max(uData, axis=1, keepdims=True)
        # y_max = np.max(yData, axis=1, keepdims=True)
        # x_max = np.max(xData, axis=1, keepdims=True)
        # uData = uData / u_max
        # yData = yData / y_max
        # xData = xData / x_max      
        # print('y_max',y_max)

        return  xData, uData ,yData
    

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
        self.q = self.m+self.p
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
            

"""DeePC class"""
            
class DeePC():
    
    def __init__(self,params_D,solver):
        
        self.H = params_D['H'] # Hankel matrix object
        self.Q = params_D['Q']
        self.R = params_D['R']
        self.solver=solver
        self.lambda_g = params_D['lambda_g'] # lambda parameter for L1 penalty norm on g
        self.r = params_D['r'] # the reference trajectory to be tracked
        self.wref=params_D['wref']
        self.E=[]
        self.E_dis=[]
        self.alpha=0.1
        self.Phi=np.block([[self.R, np.zeros((self.R.shape[0],self.Q.shape[1]))], [np.zeros((self.Q.shape[0],self.R.shape[1])), self.Q]])
        
    
    def solve_deepc(self,uini,yini):
        # COMPLETE THIS FUNCTION
        
        g=cp.Variable((self.H.T-self.H.L+1, 1))
        u=cp.Variable((self.H.m*self.H.N , 1)) 
        y=cp.Variable((self.H.p*self.H.N , 1)) 
        
        objective =  cp.sum( [ (cp.quad_form(y[self.H.p*k:self.H.p*(k+1),:]-self.r, self.Q) + cp.quad_form(u[self.H.m*k:self.H.m*(k+1),:], self.R)) for k in range(self.H.N) ] ) 
        
        
        # objective = cp.quad_form(y-r.reshape(-1,1), psd_wrap(np.kron(np.eye(N),Q))) \
        #     + cp.quad_form(u-self.uhat, psd_wrap(np.kron(np.eye(N),R))) \
        #     + lambda_1*cp.quad_form((I-Pi)@g,psd_wrap(I))\
        #     + lambda_g*cp.norm(g, 1)          
                            
        constraints = [
            self.H.Up @ g == uini,
            self.H.Yp @ g == yini,
            self.H.Uf @ g == u,
            self.H.Yf @ g == y,
        ]
        # Problem definition
        problem = cp.Problem(cp.Minimize(objective), constraints) 
        problem.solve(solver='SCS', warm_start=True)
        # print('g:',g.value)
        # print('Constraints:',self.H.Uf @ g)
        # print('u:',u.value)
        return (g.value.reshape(-1,1, order='F'), u.value.reshape(-1,1, order='F'), y.value.reshape(-1,1, order='F'))
    
    def proj(self,B,w): # w is of size Hankel.shape[0]
        B_plus = np.linalg.pinv(B)
        return B@B_plus@w
    
    def modified_davis_yin_splitting(self, w_ini, w_ref, h, tol=1e-6):
        # Initialize w, z, v
        max_iter=100
        Pi_ini=np.hstack(( np.eye(self.H.q*self.H.Tini), np.zeros((self.H.q*self.H.Tini,self.H.q*self.H.N)) )) 
        Pi_f=np.hstack(( np.zeros((self.H.q*self.H.N, self.H.q*self.H.Tini)), np.eye(self.H.q*self.H.N))) 
        w = 1*np.ones((self.H.q*self.H.L,1))
        # print(Pi_f.shape)
        # print('Pi_f@w:',Pi_f@w-w_ref)
    #     W_REF=np.vstack((wini,wref))
        for ite in range(max_iter):
            w_prev = w
            # Compute zk+1
    #         print(f"Time {ite}: \n {w[size_w*Tini:]-w_ref}")
            z = np.vstack((w_ini,Pi_f@w ))
            z_squared = np.transpose(Pi_f)@ np.kron(np.eye(self.H.N),self.Phi)@ (Pi_f@w-w_ref)

            # Compute vk+1
            v_proj=  2*z-w-2*self.alpha*z_squared
            v = self.proj(h,v_proj)
            
            # Compute wk+1
            w = w + v - z
    #         print(w-W_REF)
            
    #         e=np.linalg.norm(w[size_w*Tini:]-w_ref )
    #         e=np.transpose(w-W_REF)@ np.kron(np.eye(L),Phi) @ (w-W_REF).astype(int)
            e=np.transpose(w[self.H.q*self.H.Tini:]-w_ref)@ np.kron(np.eye(self.H.N),self.Phi) @ (w[self.H.q*self.H.Tini:]-w_ref)
            self.E.append(e)
            # Check for convergence
    #         if np.linalg.norm(w - w_prev) < tol:
    #             break

        return w
    def solve_lqr(self,uini,yini):
        u_reshape=uini.reshape(self.H.m,-1,order='F')
        y_reshape=yini.reshape(self.H.p,-1,order='F')
#         print('uini:\n',u_reshape)
#         print('yini:\n',y_reshape)
        wini=np.vstack((u_reshape,y_reshape)).reshape(-1,1, order='F')
        w = self.modified_davis_yin_splitting(wini, self.wref, self.H.Hankel)
        return w[self.H.q* self.H.Tini:]
    
    def get_next_input(self, uini, yini,s = 1):
        # Return the first control action in the predicted optimal sequence
        if self.solver=='CVXPY':
            (g, u, y) = self.solve_deepc(uini, yini) # u is u*
        elif self.solver=='lqr':
            w_f=self.solve_lqr(uini, yini)
            w_f=w_f.reshape(self.H.q,-1,order='F')
            u=w_f[:self.H.m,:].reshape(-1,1, order='F')
        elif self.solver=='NoControl':
            u=np.zeros((self.H.m*self.H.N,1))
        return u[:self.H.m*s].reshape(-1,1, order='F')  # apply (u(t),...,u(t+s))=(u_0*,...,u_s*)
    
    def get_wini(self):
        
        uini = self.H.uData[:, -(self.H.Tini):].reshape(-1, 1, order='F')
        yini = self.H.yData[:, -(self.H.Tini):].reshape(-1, 1, order='F')
        
        return uini ,yini
    
    def loop(self,Tsim,A,B,C,D,x0):
        
        usim = np.empty((self.H.m, Tsim)) # in total, Tsim simulation points
        ysim = np.empty((self.H.p, Tsim))
        uini,yini = self.get_wini()
        x = x0
        
        for t in tqdm(range(Tsim)):
            
            u = self.get_next_input(uini, yini)  # dimension (self.H.m*s, 1)
            # print(u)
            x = A@x+B@u
            y = C@x 
            usim[:, [t]] = u
            ysim[:, [t]] = y
            uini = np.block([[uini[self.H.m:]], [u]]) 
            # uini is extended with the contents of u vertically (along the rows), with the first self.H.m rows of uini being excluded. 
            yini = np.block([[yini[self.H.p:]], [y]])
        
        return usim, ysim
        

"""Distributed DeePC class"""
            
class Dis_DeePC():
    
    def __init__(self,params_D):
        
        self.H = params_D['H'] # Hankel matrix object
        self.Q = params_D['Q']
        self.R = params_D['R']
        self.lambda_g = params_D['lambda_g'] # lambda parameter for L1 penalty norm on g
        self.r = params_D['r'] # the reference trajectory to be tracked
        
    
    def solve_deepc(self,uini,yini):
        # COMPLETE THIS FUNCTION
        
        g=cp.Variable((self.H.T-self.H.L+1, 1))
        u=cp.Variable((self.H.m*self.H.N , 1)) 
        y=cp.Variable((self.H.p*self.H.N , 1)) 
        
        objective = cp.Minimize( cp.sum( [ (cp.quad_form(y[self.H.p*k:self.H.p*(k+1),:]-self.r, self.Q) + cp.quad_form(u[self.H.m*k:self.H.m*(k+1),:], self.R)) for k in range(self.H.N) ] ) )
        
        constraints = [
            self.H.Up @ g == uini,
            self.H.Yp @ g == yini,
            self.H.Uf @ g == u,
            self.H.Yf @ g == y,
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='OSQP', warm_start=True)
        # print('g:',g.value)
        # print('Constraints:',self.H.Uf @ g)
        # print('u:',u.value)
        return (g.value.reshape(-1,1, order='F'), u.value.reshape(-1,1, order='F'), y.value.reshape(-1,1, order='F'))
    
    
    def get_next_input(self, uini, yini,s = 1):
        # Return the first control action in the predicted optimal sequence
        (g, u, y) = self.solve_deepc(uini, yini) # u is u*
        return u[:self.H.m*s].reshape(-1,1, order='F')  # apply (u(t),...,u(t+s))=(u_0*,...,u_s*)
    
    def get_wini(self):
        
        uini = self.H.uData[:, -(self.H.Tini):].reshape(-1, 1, order='F')
        yini = self.H.yData[:, -(self.H.Tini):].reshape(-1, 1, order='F')
        
        return uini ,yini
    
    def loop(self,Tsim,A,B,C,D,x0):
        
        usim = np.empty((self.H.m, Tsim)) # in total, Tsim simulation points
        ysim = np.empty((self.H.p, Tsim))
        uini,yini = self.get_wini()
        x = x0
        
        for t in range(Tsim):
            
            u = self.get_next_input(uini, yini)  # dimension (self.H.m*s, 1)
            # print(u)
            x = A@x+B@u
            y = C@x +D@u
            usim[:, [t]] = u
            ysim[:, [t]] = y
            uini = np.block([[uini[self.H.m:]], [u]]) 
            # uini is extended with the contents of u vertically (along the rows), with the first self.H.m rows of uini being excluded. 
            yini = np.block([[yini[self.H.p:]], [y]])
        
        return usim, ysim
        
            
            
        
        
         
    


    
  

    
    
   
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
