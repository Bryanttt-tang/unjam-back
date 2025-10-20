import numpy as np
import time
import pickle
from tqdm import tqdm
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import ThreadPool
import concurrent.futures
import statistics
# np.random.seed(123)
class functions():
        
    def __init__(self, T, Tini, N, v, e, m, m_inter, p, M, h_total, h, connected_components, graph, alpha, max_iter, dis_iter,w_star):
        
        
        self.T = T
        self.Tini = Tini  # shift window
        self.N = N # time horizon
        self.L=self.Tini+self.N
        self.v= v
        self.e= e
        self.m_inter=m_inter # number of interconnected inputs of each unit
        self.m=m # num_input of each unit
        self.m_total= m*self.v # total manifest variable
        self.m_dis = self.m_total+self.e*self.m_inter
        self.p=p # num_output of each unit
        self.p_total=p*self.v
        self.q=self.m_total+self.p_total
        self.q_dis=self.m_dis+self.p_total
        self.E=[]
        self.E1=[]
        self.M=M
        self.w_star=w_star
        self.M_inv=np.linalg.inv(self.M@self.M.T)  
        self.connected_components=connected_components 
        self.graph=graph
        self.num_neighbor=[]
        for i in range(self.v):
            self.num_neighbor.append( len(list(graph.neighbors(i))) )
        # print('neighbors:\n',self.num_neighbor)
        self.alpha=alpha
        self.max_iter=max_iter
        self.dis_iter=dis_iter
        self.k_lqr=[]
        self.k_dis_lqr=[]
        self.E_dis=[]
        self.time_lqr=[]
        self.time_proj=[]
        self.time_worst=[]
        self.worst_sub=[]
        self.time_alter_proj=[]
        self.time_dis_lqr=[]
        self.time_proj2=[]
        self.time_sub=[]
        self.time_thread=[]
        self.time_split=[]
        self.time_split2=[]
        self.time_inter=[]
        self.all_sub=[]
        self.h_total=h_total # a matrix
        self.h=h # h is a list
        U, S, VT = np.linalg.svd(self.h_total)
        self.rank_total = self.m_total*self.L+2*self.v
        print('rank',self.rank_total)
        self.U_truncated = U[:, :self.rank_total]  # Shape (48, 27)
        self.proj_h=self.U_truncated@np.linalg.inv(self.U_truncated.T@self.U_truncated)@self.U_truncated.T
        start_lqr_off = time.process_time()
        # self.proj_h=self.h_total@np.linalg.pinv(self.h_total)
        end_lqr_off = time.process_time()
        self.lqr_off_time=end_lqr_off-start_lqr_off

        start_dislqr_off = time.process_time()
        self.proj_h_sub=[]
        for k in range(len(self.h)):
            self.proj_h_sub.append(self.h[k]@np.linalg.pinv(self.h[k]))
        end_dislqr_off = time.process_time()
        self.dislqr_off_time=end_dislqr_off-start_dislqr_off
    def matrix_vector_multiply(self,matrix, vector):
        """
        Multiplies a matrix by a vector.

        :param matrix: A numpy array of shape (m, n) where each row is a list in the matrix.
        :param vector: A numpy array of shape (n,) representing the vector.
        :return: A numpy array of shape (m,) representing the resulting vector.
        """
        rows, cols = matrix.shape
        
        if vector.shape[0] != cols:
            raise ValueError("The number of columns in the matrix must equal the number of elements in the vector")
        
        result = np.zeros(rows)
        
        for i in range(rows):
            for j in range(cols):
                result[i] += matrix[i, j] * vector[j]
        
        return result.reshape(-1,1)
    
    def proj(self,B,w): # w is of size Hankel.shape[0]
        B_plus = np.linalg.pinv(B)
        return B@B_plus@w
    #     return B@np.linalg.inv(np.transpose(B)@B)@np.transpose(B)@w

    def timed_matmul(self,x, y):
        start_time = time.process_time()
        result = x @ y
        # result = self.matrix_vector_multiply(x,y)
        # result = self.proj(x,y)
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    
    def proj_two(self,h,w,pool): # Projection onto the Cartisian Product of ten subspace

        w_re=w.reshape(self.q_dis,-1,order='F')
        w_reorder=[]
        start_split = time.process_time()
        start_row=0
        for i in range(self.v):
        
            num_rows = self.m + self.num_neighbor[i]*self.m_inter
            end_row = start_row + num_rows
            rows=np.r_[start_row:end_row, (self.q_dis - (self.p_total - i)) : (self.q_dis - (self.p_total - i)+self.p)]
            w_reorder.append(w_re[rows,:].reshape(-1,1,order='F'))
            start_row = end_row
        end_split = time.process_time()
        self.time_split.append(end_split-start_split)
        
        # start_thread = time.process_time()
        # with ThreadPool(processes=16) as pool:
            # results=pool.starmap(lambda x, y: x @ y, zip(h, w_reorder))
        results=pool.starmap(self.timed_matmul, zip(h, w_reorder))
            #  results = pool.starmap(self.proj, [(h[i], w_reorder[i]) for i in range(10)])
        w_proj=[result[0] for result in results]    
        self.time_sub=[result[1] for result in results]
        worst_sub=np.max(self.time_sub)
        self.worst_sub.append(worst_sub) # get the worst subspace in each iteration
        mean_sub=statistics.mean(self.time_sub)
        var_sub=statistics.variance(self.time_sub)
        self.all_sub.append(self.time_sub)
        # end_thread = time.process_time()
        # self.time_thread.append(end_thread-start_thread)
        # w_proj = results

        result_vectors = np.zeros((self.q_dis, len(w)//self.q_dis))
        start_split2 = time.process_time()
        start_row2=0
        for i in range(self.v):
            num_rows = self.m + self.num_neighbor[i]*self.m_inter
            end_row2 = start_row2 + num_rows
            rows=np.r_[start_row2:end_row2, (self.q_dis - (self.p_total - i)) : (self.q_dis - (self.p_total - i)+self.p)]
            # rows=np.r_[np.arange(self.m+1+(self.m+2)*(i-1), self.m_dis), self.q_dis - (self.p_total - i)]
            result_vectors[rows,:]=w_proj[i].reshape( (self.m+self.num_neighbor[i]*self.m_inter+self.p) ,-1,order='F')
            start_row2 = end_row2
        end_split2 = time.process_time()
        self.time_split2.append(end_split2-start_split2)
        self.time_worst.append(end_split2-start_split2 + end_split-start_split + worst_sub)       
        return result_vectors.reshape(-1,1,order='F')
    
    # def proj_full(self,h,w): # Projection onto the Cartisian Product of ten subspace
    #     w_re=w.reshape(self.q_dis,-1,order='F')
    #     w_reorder=[]
    #     start_split = time.process_time()
    #     for i in range(self.v):
    #         rows=np.r_[np.arange((self.m+self.v-1)*i, (self.m+self.v-1)*(i+1)), self.q_dis - (self.p_total - i)]
    #         w_reorder.append(w_re[rows,:].reshape(-1,1,order='F') )
    #     end_split = time.process_time()
    #     self.time_split.append(end_split-start_split)
    #     start_thread = time.process_time()
    #     with ThreadPool(processes=8) as pool:
    #         # results=pool.starmap(lambda x, y: x @ y, zip(h, w_reorder))
    #         results=pool.starmap(self.timed_matmul, zip(h, w_reorder))
    #         #  results = pool.starmap(self.proj, [(h[i], w_reorder[i]) for i in range(10)])
    #     w_proj=[result[0] for result in results]    
    #     self.time_sub=[result[1] for result in results]
    #     # w_proj = results
    #     end_thread = time.process_time()
    #     self.time_thread.append(end_thread-start_thread)
        
    #     result_vectors = np.zeros((self.q_dis, len(w)//self.q_dis))
    #     start_split2 = time.process_time()
    #     for i in range(self.v):
    #         rows=np.r_[np.arange((self.m+self.v-1)*i, (self.m+self.v-1)*(i+1)), self.q_dis - (self.p_total - i)]
    #         result_vectors[rows,:]=w_proj[i].reshape((self.m+self.v-1+self.p),-1,order='F')
    #     end_split2 = time.process_time()
    #     self.time_split2.append(end_split2-start_split2)          
    #     return result_vectors.reshape(-1,1,order='F')
                

    # def proj_inter(self,M,M_inv,w): # Projection onto interconnected  behavioral subspace
    #     Tf=len(w)//self.q_dis
    # #     print(length)
    # #     print(ker.shape)
    # #     ker=np.hstack([A] * length)
    # #     return w-1/(2*length)*np.transpose(ker)@ker@w
    #     return w-np.kron(np.eye(Tf),M.T@M_inv@M) @w
    
    def custom_mean(self, data, row_indices):
        num_columns = len(data[0])
        sum_row = [0] * num_columns
        for i in row_indices:
            if len(data[i]) != num_columns:
                raise ValueError(f"Row {i} does not have the correct number of columns.")
            sum_row = [sum_row[j] + data[i][j] for j in range(num_columns)]
        
        mean_row = [x / len(row_indices) for x in sum_row]
        return np.array(mean_row)
    
    def proj_inter_2(self,w): # Projection onto interconnected  behavioral subspace
        w_re=w.reshape(self.q_dis,-1,order='F') # O(1)
        for group in self.connected_components:
            # w_re[group,:] = self.custom_mean(w_re,group)
            w_re[group,:] = np.mean(w_re[group,:],axis=0)
        return w_re.reshape(-1,1,order='F')
    
    def project_onto_box_constraints(self,w):
        # w_projected = w.copy()
        w_re=np.copy(w).reshape(-1,self.L,order='F')
        w_re[:self.m_total, self.Tini:] = np.clip(w_re[:self.m_total, self.Tini:], -0.5, 0.5)  # Apply box constraints to the inputs of each column
        # print('w_re',w_re)
        # print(w_re.shape)
        return w_re.reshape(-1,1,order='F')
    
    def alternating_projections(self,h,x, pool, num_iterations=10, tol=1e-10):
        for _ in range(num_iterations):
            # start_proj2=time.process_time()
            # x = self.proj_full(h,x) # 2*O(n)+ m*O(n_small^2)/8, where m is the number of units
            x = self.proj_two(h,x,pool) # 2*O(n)+ m*O(n_small^2)/8, where m is the number of units
            # end_proj2=time.process_time()
            # self.time_proj2.append(end_proj2-start_proj2)
            
            start_inter = time.process_time()
            x = self.proj_inter_2(x) # O(n)
            end_inter = time.process_time()
            self.time_inter.append(end_inter-start_inter)
            # x = self.proj_inter(M,M_inv,x)
            # Check convergence
    #         if np.linalg.norm(x - proj_square(x)) < tol:
    #             break
        return x

    def alternating_projections2(self,h,x, num_iterations=10, tol=1e-10):
        x_copy=x.copy()
        for _ in range(num_iterations):
            x_copy=self.proj_h @x_copy
            x_copy=self.project_onto_box_constraints(x_copy)
            # x = self.proj_inter(M,M_inv,x)
            # Check convergence
    #         if np.linalg.norm(x - proj_square(x)) < tol:
    #             break
        return x_copy
    
    def average_projections(self,h,M,M_inv,x, num_iterations=150, tol=1e-10):
        for _ in range(num_iterations):
            x1 = self.proj_two(h,x)
            x2 = self.proj_inter(M,M_inv,x)
            x=(x1+x2)/2
            # Check convergence
    #         if np.linalg.norm(x - proj_square(x)) < tol:
    #             break
        return x

    # Dykstra's Alternating Projections Algorithm
    def dykstra_alternating_projections(self,h,M,M_inv,x, num_iterations=50, tol=1e-6):
        y = np.copy(x)
        p = np.zeros_like(x)
        q = np.zeros_like(x)
        for _ in range(num_iterations):
    #         y_old = np.copy(y)
            # Update y
            y = self.proj_two(h,y + p)
    #         y = proj_square(y)
            # Update p
            p = x + p - y
            # Update x
    #         x = proj_circle(y + q)
            x = self.proj_inter(M,M_inv,y+q)
    #         x = proj(A,y+q)
            # Update q
            q = y + q - x
            # Check convergence
    #         if np.linalg.norm(y - y_old) < tol:
    #             break
        return x
    
    def lqr(self,w_ini, w_ref, Phi, tol=1e-8): # Alberto's algorithm
        # Initialize w, z, v
        # w=np.vstack((w_ini, w_ref ))
        w = np.zeros((self.q*self.L,1))
        # w = 10*np.random.rand(self.q*self.L,1)-5
        kron=np.diag( np.kron(np.eye(self.N),Phi) ).reshape(-1, 1) # a vector containing all diagonal elements
        e=np.dot((w[self.q*self.Tini:]-w_ref).T, (kron * (w[self.q*self.Tini:]-w_ref)))[0,0]
        # e=np.dot( (w-np.vstack((w_ini,w_ref))).T, (w-np.vstack((w_ini,w_ref))))[0,0]
        # e1=np.linalg.norm(self.w_star - w[self.q*self.Tini:])/np.linalg.norm(self.w_star)
        self.E.append(e)
        # self.E1.append(e1)
        k=0
        for ite in range(self.max_iter):
            # w_prev = w
            # Compute zk+1
            start_lqr=time.process_time()
            z = np.vstack((w_ini,w[-self.q*self.N:] )) # O(n), n=q*(Tini+Tf)
    #         z_squared = np.transpose(Pi_f)@  kron* (w[-q*N:]-w_ref)
            z_squared = np.vstack(( np.zeros((self.q*self.Tini,1)) ,kron * (w[-self.q*self.N:]-w_ref) )) #O(2n)

            # Compute vk+1
            v_proj=  2*z-w-2*self.alpha*z_squared # O(n)
            # print('v_proj:',v_proj.shape)
            start=time.process_time()
            # v_plus = self.proj_h @ v_proj # O(n^2)
            v_plus = self.alternating_projections2(self.proj_h_sub, v_proj, num_iterations=self.dis_iter) 
            # v_plus = self.matrix_vector_multiply(self.proj_h, v_proj) # O(n^2)
            # print('v_plus',v_plus.shape)
            end=time.process_time()
            self.time_proj.append(end-start)
            # v_plus = self.proj(h, v_proj)
            
            # Compute wk+1
            w = w + v_plus - z # O(n)
            
            end_lqr=time.process_time()
            self.time_lqr.append(end_lqr-start_lqr)
            e=np.dot((w[self.q*self.Tini:]-w_ref).T, (kron * (w[self.q*self.Tini:]-w_ref)))[0,0]
            # e1=np.linalg.norm(self.w_star - w[self.q*self.Tini:])/np.linalg.norm(self.w_star)
            # e=np.dot( (w-np.vstack((w_ini,w_ref))).T, (w-np.vstack((w_ini,w_ref))))[0,0]
            self.E.append(e)
            # self.E1.append(e1)
            # Check for convergence
            k+=1
            # # # print( 'norm',np.linalg.norm(w - w_prev))
            # if np.linalg.norm(w - w_prev) < tol:
            #     break
        self.k_lqr.append(k)
        return w

    def distributed_lqr(self, w_ini, w_ref, Phi, tol=1e-7):
        # Initialize w, z, v
        with ThreadPool(processes=16) as pool:
            # w=np.vstack((w_ini, np.zeros((self.q_dis*self.N,1)) ))
            w = np.zeros((self.q_dis*self.L,1))
            # w=np.vstack((w_ini, w_ref ))
            kron=np.diag( np.kron(np.eye(self.N),Phi) ).reshape(-1, 1)
            e=np.dot((w[self.q_dis*self.Tini:]-w_ref).T, (kron * (w[self.q_dis*self.Tini:]-w_ref)))[0,0]
            # e=np.dot( (w-np.vstack((w_ini,w_ref))).T, (w-np.vstack((w_ini,w_ref))))[0,0]
            self.E_dis.append(e)
            k=0
            for ite_dis in tqdm(range(self.max_iter)):
                w_prev = w
                # Compute zk+1
                start_dislqr=time.process_time()
                z = np.vstack((w_ini,w[-self.q_dis*self.N:] ))
        #         z_squared = np.transpose(Pi_f)@ kron * (w[-q*N:]-w_ref)
                
                z_squared = np.vstack(( np.zeros((self.q_dis*self.Tini,1)) ,kron * (w[-self.q_dis*self.N:]-w_ref) ))
            
                # Compute vk+1
                v_proj=  2*z-w-2*self.alpha*z_squared
                start=time.process_time()
                # if ite_dis<=self.max_iter-20:
                #     v_plus = self.alternating_projections(self.proj_h_sub, v_proj, pool, num_iterations=1) 
                # else:
                #     v_plus = self.alternating_projections(self.proj_h_sub, v_proj, pool, num_iterations=self.dis_iter) 
                v_plus = self.alternating_projections(self.proj_h_sub, v_proj, pool, num_iterations=self.dis_iter) 
                end=time.process_time()
                self.time_alter_proj.append(end-start)
                # v_plus = self.alternating_projections(h, self.M, self.M_inv, v_proj, num_iterations=self.dis_iter) 
                # Compute wk+1
                w = w + v_plus - z 
                end_dislqr=time.process_time()
                self.time_dis_lqr.append(end_dislqr-start_dislqr)
                
                e=np.dot((w[self.q_dis*self.Tini:]-w_ref).T, (kron * (w[self.q_dis*self.Tini:]-w_ref)))[0,0]
                # e=np.dot( (w-np.vstack((w_ini,w_ref))).T, (w-np.vstack((w_ini,w_ref))))[0,0]
                self.E_dis.append(e)
                # Check for convergence
                # k+=1
                if np.linalg.norm(w - w_prev) < tol:
                    break
            self.k_dis_lqr.append(k)
        return w