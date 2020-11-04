import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, LinearOperator
import inspect  #for checking signatures of functions

"""
NUMERICAL METHOD, USED IN IMPLICIT RENORMALIZATION METHOD IN ELIASHBERG CLASS DEFINED BELOW 
"""
#solve the equation |v> = A|v> + |b> for |v>
#method of Prokof'ev and Svistunov from PRL 99, 250201 (2007)
def solve_linear(A,b,max_iter=500,threshold=0.001,v_init = None):

    if v_init is None:
        v_init = np.random.rand(len(b))

    #initialize vectors
    v_old = v_init
    v_avg = v_old
    v_new = np.empty(len(b))
    v_output = np.empty(len(b))

    i =  1
    while True:
        #find v_{i+1}
        v_new[:] = np.matmul(A,v_avg)+b
        #check if v_{i+1} and v_i are close (v has converged)
        if np.linalg.norm(v_new-v_old) < threshold:
            v_output[:] = v_new
            print('convergence of vector solution achieved')
            print('number of iterations:',i)
            break
        else:
            if i > max_iter:
                v_output[:] = v_old
                print('convergence of vector solution not achieved')
                print('diff is',np.linalg.norm(v_new-v_old))
                break
            #find v_avg(i+1), which will be used for the next i
            v_avg[:] = (v_avg*i + v_new)/(i+1)
             #update v_{i+1}
            v_old[:] = v_new
            i += 1
    return v_output

"""
FUNCTIONS FOR MATSUBARA FREQUENCIES
"""

def matsubara(idx,temp):
    return (2*idx+1)*np.pi*temp

def mf_list(temp,freq_cut):
    npts = 2*int(freq_cut/(2*np.pi*temp))

    if npts > 2:
        return matsubara(np.arange(npts)-npts/2,temp)
    else:
        raise Exception('number of matsubara frequencies is less than 2. increase freq_cut.')

"""
DEFINITION OF THE ELIASHBERG CLASS, WHICH CAN BE USED TO FIND Tc
"""

class Eberg:
    def __init__(self,pot_fn,freq_cut, en_list, weight_list, mu, dos):
        
        #check that pot_fn is a function that takes in 4 arguments
        if len(inspect.getfullargspec(pot_fn)[0]) != 4:
            raise TypeError('The interaction potential must take in 4 arguments: {matsubara frequency 1, energy 1, matsubara frequency 2, energy 2}')
       
       #if freq_cut is not a number, or is a boolean, raise type error 
        if not isinstance(freq_cut,(int,float)) or isinstance(freq_cut,bool):
            raise TypeError('freq_cut must be a number!')
        #if freq_cut is a negative number, raise value error
        if freq_cut <= 0:
            raise ValueError('freq_cut must be positive!')

        #check that en_list and weight_list are lists or numpy arrays
        if not isinstance(en_list,(list,np.ndarray)):
            raise TypeError('en_list must be a list!')
        if not isinstance(weight_list,(list,np.ndarray)):
            raise TypeError('weight_list must be a list!')

        #check that en_list and weight_list have the same length
        if len(en_list) != len(weight_list):
            raise TypeError('en_list and weight_list must be the same length!')

        #check that en_list and weight_list are lists of floats
        if all(isinstance(x,float) for x in en_list) == False:
            raise TypeError('en_list must be a list of only floats!')
        if all(isinstance(x,float) for x in weight_list) == False:
            raise TypeError('weight_list must be a list of only floats!')

        #if mu is not a number, or is a boolean, raise type error 
        if not isinstance(mu,(int,float)) or isinstance(mu,bool):
            raise TypeError('mu must be a real number!')

        
        #check that dos is a function that takes in 1 argument
        if len(inspect.getfullargspec(dos)[0]) != 1:
            raise TypeError('The dos function must take in 1 argument: energy')

        #check that dos is a function that gives non-negative values at all energies
        for en in en_list:
            #first check if dos(en) is a number
            if not isinstance(dos(en),float):
                raise TypeError('the density of states must be a function that returns floats!')
            if dos(en) < 0:
                raise ValueError('the density of states must return non-negative values!')

        self.pot_fn = pot_fn
        self.freq_cut = freq_cut
        self.en_list = en_list
        self.weight_list = weight_list
        self.mu = mu
        self.dos = dos
        
    def num_freq(self,temp):
        num_mf = 2*int(self.freq_cut/(2*np.pi*temp))
   
        if num_mf > 2:
            return num_mf
        else:
            raise Exception('number of matsubara frequencies is less than 2. increase freq_cut or decrease temperature.')

    def num_energies(self):
        return len(self.en_list)
        
    def vec_size(self,temp):
        return self.num_freq(temp) * self.num_energies()
   
    #does the same thing as mf_list function defined above
    def mat_freq_list(self,temp):
        npts = self.num_freq(temp)
        return matsubara(np.arange(npts)-npts/2,temp)
    
    def find_Z(self,temp,verbose = 0, self_consistent = True):
        max_iter = 100
        tol = 0.001
        
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        
        #initalize Z arrays 
        Z_in = np.ones((num_mf,num_en))
        Z_out= np.empty((num_mf,num_en))
        
        #define arrays which will be used to iterate Z and phi
        mf_list = self.mat_freq_list(temp)
        interaction_array = self.pot_fn(mf_list[:,None,None,None],self.en_list[None,:,None,None],mf_list[None,None,:,None],self.en_list[None,None,None,:])
       
        if verbose == 1:
            print('Size of Z:',npts)
        
        for idx in range(max_iter):
            
            #iterate Z 
            Z_out[:,:] = 1.0 - temp/mf_list[:,None] * np.einsum('ijkl,kl',interaction_array,\
                    self.weight_list[None,:]*self.dos(self.en_list[None,:]) * mf_list[:,None] * Z_in \
                    /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu,2)))
            
            
            
            diff = np.linalg.norm(Z_out - Z_in)
            
            if verbose == 1:
                print('\nIteration:',idx)
                print('diff:',diff)

            if self_consistent == False and idx == 1:
                return Z_out
            
            if  diff < tol:
                if verbose == 1:
                    print('\nConvergence achieved')
                    print('diff:',diff)
                return Z_out
            
            Z_in[:,:] = Z_out
        
        #if we have reached this point, convergence has not been achieved
        if verbose == 1:
            print('Convergence has not been achieved')
        return Z_out
    
    def find_Z_chi(self,temp,verbose = 0, self_consistent = True):
        max_iter = 100
        tol = 0.001
        
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        
        #initalize Z arrays 
        Z_in = np.ones((num_mf,num_en))
        Z_out= np.empty((num_mf,num_en))
        
        #initalize chi arrays 
        chi_in = np.zeros((num_mf,num_en))
        chi_out= np.empty((num_mf,num_en))
        
        #define arrays which will be used to iterate Z and phi
        mf_list = self.mat_freq_list(temp)
        interaction_array = self.pot_fn(mf_list[:,None,None,None],self.en_list[None,:,None,None],mf_list[None,None,:,None],self.en_list[None,None,None,:])
       
        if verbose == 1:
            print('Size of Z:',num_en * num_mf)
        
        for idx in range(max_iter):
            
            #iterate Z 
            Z_out[:,:] = 1.0 - temp/mf_list[:,None] * np.einsum('ijkl,kl',interaction_array,\
                    self.weight_list[None,:]*self.dos(self.en_list[None,:]) * mf_list[:,None] * Z_in \
                    /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu + chi_in,2)))
            
            #iterate chi 
            chi_out[:,:] =  temp * np.einsum('ijkl,kl',interaction_array,\
                    self.weight_list[None,:]*self.dos(self.en_list[None,:]) * (self.en_list[None,:]-self.mu + chi_in) \
                    /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu + chi_in,2)))
            
            
            diff = np.linalg.norm(Z_out - Z_in) + np.linalg.norm(chi_out - chi_in)
            
            if verbose == 1:
                print('\nIteration:',idx)
                print('diff:',diff)

            if self_consistent == False and idx == 1:
                return (Z_out,chi_out)
            
            if  diff < tol:
                if verbose == 1:
                    print('\nConvergence achieved')
                    print('diff:',diff)
                return (Z_out,chi_out)
            
            Z_in[:,:] = Z_out
            chi_in[:,:] = chi_out
        
        #if we have reached this point, convergence has not been achieved
        if verbose == 1:
            print('Convergence has not been achieved')
        return (Z_out,chi_out)

    def find_kernel(self,temp,verbose = 0,include_chi = False):

        if not isinstance(include_chi,bool):
            raise TypeError('include_chi must be a bool!')

        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
        
        if verbose == 1:
            print('\nKernel Size:',npts,'x',npts)
       
        if include_chi == True:
            Z,chi = self.find_Z_chi(temp,verbose)
       
        else:
            Z = self.find_Z(temp,verbose)

        mf_list = self.mat_freq_list(temp)
        
        
        #reshape lists for easy broadcasting
        w_list_shaped = self.weight_list[None,:]
        en_list_shaped = self.en_list[None,:]
        dos_list_shaped = self.dos(en_list_shaped)
        mf_list_shaped = mf_list[:,None]
        

        #try new find of finding kernel
        kernel = np.empty((num_mf,num_en,num_mf,num_en))

        if include_chi == True:

            kernel = -1.0 * temp * self.pot_fn(mf_list[:,None,None,None],self.en_list[None,:,None,None],mf_list[None,None,:,None],self.en_list[None,None,None,:])\
                    * self.weight_list[None,None,None,:] * self.dos(self.en_list[None,None,None,:]) \
                    / (np.power(mf_list[None,None,:,None] * Z[None,None,:,:],2) + np.power(self.en_list[None,None,None,:]-self.mu + chi,2))

        else:
            
            kernel = -1.0 * temp * self.pot_fn(mf_list[:,None,None,None],self.en_list[None,:,None,None],mf_list[None,None,:,None],self.en_list[None,None,None,:])\
                    * self.weight_list[None,None,None,:] * self.dos(self.en_list[None,None,None,:]) \
                    / (np.power(mf_list[None,None,:,None] * Z[None,None,:,:],2) + np.power(self.en_list[None,None,None,:]-self.mu,2))

        kernel = np.reshape(kernel,(num_mf * num_en,num_mf * num_en))

        if verbose == 1:
            print('Kernel found')
            
        return kernel
    

    def find_kernel_sorted(self,temp,om_cut = 1.0, verbose = 0):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
        
        if verbose == 1:
            print('\nKernel Size:',npts,'x',npts)
        
        Z = self.find_Z(temp,verbose)
        Z = np.reshape(Z,npts)
                
        mf_list = self.mat_freq_list(temp)
        
        #make long list for en_tot
        en_tot_list = np.sqrt(np.power(mf_list[:,None],2) + np.power(self.en_list-self.mu,2)).reshape(npts)
        idx_sort_list = np.argsort(en_tot_list)
        
        mf_idx_sort_list, en_idx_sort_list = np.divmod(idx_sort_list,num_en)
                
        #use the idx_sort_list to sort the en_list_long and mf_list_long arrays
        mf_list_sorted = mf_list[mf_idx_sort_list]
        en_list_sorted = self.en_list[en_idx_sort_list]
        weight_list_sorted = self.weight_list[en_idx_sort_list]
        dos_list_sorted = self.dos(en_list_sorted)
        Z_sorted = Z[idx_sort_list]

#        #use above lists to construct sorted kernel
#        kernel = np.empty((npts,npts))
#        for idx in range(npts):
#            kernel[idx,:] =  -1.0 * temp \
#            * self.pot_fn(mf_list_sorted[idx],en_list_sorted[idx],mf_list_sorted,en_list_sorted)\
#            * weight_list_sorted * dos_list_sorted\
#            / (np.power(Z_sorted * mf_list_sorted,2) + np.power(en_list_sorted - self.mu,2))
   
        #construct sorted kernel, in new (hopefully faster) way
        kernel = np.empty((npts,npts))
        kernel = -1.0 * temp \
                * self.pot_fn(mf_list_sorted[:,None],en_list_sorted[:,None],mf_list_sorted[None,:],en_list_sorted[None,:])\
                * weight_list_sorted[None,:] * dos_list_sorted[None,:]\
                / (np.power(Z_sorted[None,:] * mf_list_sorted[None,:],2) + np.power(en_list_sorted[None,:] - self.mu,2))


        return kernel
    
    """ONLY FOR IMPLICIT RENORMALIZATION. APPROPRIATE ONLY WHEN THERE IS A COOPER LOG"""
    def find_lambda_bar(self,temp,om_cut,return_gap = False):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
                
        mf_list = self.mat_freq_list(temp)
        
        en_tot_list = np.sqrt(np.power(mf_list[:,None],2) + np.power(self.en_list-self.mu,2)).reshape(npts)
        idx_sort_list = np.argsort(en_tot_list)
        
        en_tot_sorted = en_tot_list[idx_sort_list]
        
        #find idx_bd that separates "low" energies from "high" energies
        idx_bd = None
        for idx in range(npts):
            if en_tot_sorted[idx] > om_cut:
                idx_bd = idx
                print('\nidx_bd:',idx_bd,' out of ',npts)
                break
        if idx_bd == None:
            raise ValueError('Choose a smaller om_cut! Om_cut must be smaller than the energy cutoffs to ensure a separation between low and high energies!')


                
        #use this idx_bd to break up kernel
        #ADD EXTRA MINUS SIGN SO THAT THE KERNEL IS CONSISTENT WITH THE CONVENTION OF
        #THE IMPLICIT RENORMALIZATION PAPER 
        kernel = -1.0 * self.find_kernel_sorted(temp)
        
        K11 = kernel[:idx_bd, :idx_bd]
        K12 = kernel[:idx_bd, idx_bd:]
        K21 = kernel[idx_bd:, :idx_bd]
        K22 = kernel[idx_bd:, idx_bd:]
        
        #use these to find lambda_bar
        phi = np.random.rand(npts)
        
        phi1 = phi[:idx_bd]
        phi2 = phi[idx_bd:]
        
        phi1_new = np.empty(len(phi1))
        
        counter = 0
        threshold = 0.0001
        max_count = 100
        
        lambda_bar = 0.0
        
        
        while True:


            phi2[:] = solve_linear(-1.0*K22,np.matmul(-1.0*K21,phi1),max_iter=500,v_init=phi2)

            #rewriting of vector eqn \lambda phi_1 = - B phi_1
            #phi1_new is \lambda phi_1, ideally proportional to phi_1 if input phi_1 is eigenvector of B
            phi1_new[:] = -1*np.matmul(K11,phi1)-np.matmul(K12,phi2)
            lambda_bar_new = np.dot(phi1,phi1_new)/np.dot(phi1,phi1)
            phi1_new[:] = phi1_new/lambda_bar_new
            diff = np.linalg.norm(phi1_new-phi1)
            if diff < threshold and np.abs(lambda_bar - lambda_bar_new) < 0.001:
                #print('convergence of phi1 has been achieved')
                #print('diff of phi1:',diff)
                #convergence has been achieved. output lambda_bar
                break
            else:
                counter += 1
                phi1[:] = phi1_new
                lambda_bar = lambda_bar_new

            counter += 1

            if counter > max_count:
                print('convergence of phi1 has not been achieved')
                print('diff of phi1 is',diff)
                break
        print('lambda_bar:',lambda_bar)
        
        if return_gap == False:
            return lambda_bar_new
        
        else:
            #rescale lower frequency phi by lambda_bar
            phi1[:] = phi1_new*lambda_bar_new
            
            del phi1_new
    
            #unsort phi
            phi_unsorted = np.empty(npts)
            for i in range(npts):
                idx = idx_sort_list[i]
                phi_unsorted[idx] = phi[i]
            del phi
            
            return np.reshape(phi_unsorted,(-1,num_en))
            
#             """THINK ABOUT HOW TO INCLUDE Z. DOES Z CHANGE WITH TEMP SUBSTANTIALLY?
#             MAYBE IT DOESN'T IN THE LOW T LIMIT. AT WEAK COUPLING THEN, WE CAN TAKE Z
#             AT T = 0, I.E. NOT WORRY ABOUT HOW Z CHANGES WITH T"""
#             Z = self.find_Z(temp)
            
#             #return gap
#             return phi_unsorted/Z
                    
    """Method to find gap. Temp must be small enough to be in the large logarithm regime.
    In this limit, gap should not depend on temp"""
    def find_gap_ir(self,temp,om_cut = 1.0):
        """All of the work in finding the gap is done while finding lambda_bar.
        Therefore, we simply point to that function here."""
        return self.find_lambda_bar(temp,om_cut,return_gap = True)
    
    def __find_eigval(self,temp,verbose = False, include_chi = False):
        kernel = self.find_kernel(temp,verbose = verbose, include_chi = include_chi)
        return np.real(eigs(kernel,k=1,which='LR')[0])[0]
    
    def find_tc_eigval(self,tc0=0.1,tol = 0.0001,max_iter = 10,include_chi = False):
        temp_list = []
        eigval_list = []
        
        temp_1 = tc0
        eigval_1 = self.__find_eigval(temp_1,include_chi = include_chi)
        #bool_1 = eigval_1 > 1
        
        print('temp:',temp_1)
        print('eigval:',eigval_1)
       

        #define temp_2 to be closer to T_c
        
        #if eigval_1 > 1, then temp_1 < T_c
        if eigval_1 > 1:
            temp_2 = temp_1 * 2.0
            eigval_2 = self.__find_eigval(temp_2,include_chi = include_chi)
        else:
            temp_2 = temp_1 * 0.5
            eigval_2 = self.__find_eigval(temp_2,include_chi = include_chi)

        
        print('temp:',temp_2)
        print('eigval:',eigval_2)
        
        temp_list.append(temp_1)
        temp_list.append(temp_2)
        eigval_list.append(eigval_1)
        eigval_list.append(eigval_2)
        
        
        
        for num in range(max_iter):
            #if temp's are both still larger than T_c or both less than T_c, iterate temp's
            if (eigval_1 > 1) == (eigval_2 > 1):
                
                #reassign temp_2 to temp_1
                #reassign eigval_2 to eigval_1
                temp_1 = temp_2
                eigval_1 = eigval_2
               
                #define temp_2 to be closer to T_c
                
                #if eigval_1 > 1, then temp_1 < T_c
                if eigval_1 > 1:
                    temp_2 = temp_1 * 2.0
                    eigval_2 = self.__find_eigval(temp_2,include_chi = include_chi)
                    print('temp:',temp_2)
                    print('eigval:',eigval_2)
                else:
                    temp_2 = temp_1 * 0.5
                    eigval_2 = self.__find_eigval(temp_2,include_chi = include_chi)
                    print('temp:',temp_2)
                    print('eigval:',eigval_2)
                
                
                temp_list.append(temp_2)
                eigval_list.append(eigval_2)
    
            else:
                print("Signs are different! We have a range between T_c's")
                break
               
        #after max iterations, we may not have 2 temp's that bound T_c
        #if so, just return temp_2 as the closest temp to T_c
        if eigval_1 >1 == eigval_2 > 1:
            print("Newton's method not applied. Have not bounded Tc between 2 values")
            
            if eigval_2 > 1:
                print("Lower bound on T_c:",temp_2)
            else:
                print("Upper bound on T_c:",temp_2)
            return temp_2
      
        #define bools for use below
        bool_1 = eigval_1 > 1
        bool_2 = eigval_2 > 1

        #define variables in preparation for Newton's method
        m = (eigval_2-eigval_1)/(temp_2-temp_1)
        temp = temp_1 + (1-eigval_1)/m
        eigval = self.__find_eigval(temp,include_chi = include_chi)
        sign  = eigval > 1
        
        #Here begins Newton's method
        for i in range(max_iter):
            print('new temp:',temp)
            print('new eigval',eigval)
            
            #Check for convergence
            if np.abs(eigval-1) < tol:
                temp_eigval_array = np.array([temp_list,eigval_list])
                np.savetxt('temp_lambda.txt',np.transpose(temp_eigval_array))
                
                return temp
        
                
            print("\nIteration:",i)
            if sign == bool_1:
                temp_1 = temp
                eigval_1 = eigval
                bool_1 = sign
                
            else:
                temp_2 = temp
                eigval_2 = eigval
                bool_2 = sign
                
                
            m = (eigval_2 - eigval_1)/(temp_2-temp_1)
            temp = temp_1 + (1-eigval_1)/m
            eigval = self.__find_eigval(temp,include_chi = include_chi)
            sign  = eigval > 1
            
            
            temp_list.append(temp)
            eigval_list.append(eigval)
                
        #If here, we have gone through max_iter iterations of Newton's method
        #check for convergence
        if np.abs(eigval-1) < tol:
            temp_eigval_array = np.array([temp_list,eigval_list])
            np.savetxt('temp_lambda.txt',np.transpose(temp_eigval_array))
            
            print('\nTc has been found!')
            print('tc:',temp)
            return temp
                
        else:
            temp_eigval_array = np.array([temp_list,eigval_list])
            np.savetxt('temp_lambda.txt',np.transpose(temp_eigval_array))
            print('convergence not found!')
            print('best approximation of temp:',temp)
            return temp
        
    def find_gap(self,tc=None,tc0 = 0.1,include_chi = False):

        print('\nFINDING GAP\n')

        #check if tc is input. If not, find tc
        if tc is None:
            tc = self.find_tc_eigval(tc0,include_chi = include_chi)


        print('\ntc:',tc)
        e_val,e_vec = eigs(self.find_kernel(tc,include_chi = include_chi),k=1,which='LR')
        phi = np.real(e_vec)
        
        if include_chi == True:
            Z = self.find_Z(tc)
        else:
            Z,chi = self.find_Z_chi(tc)

        gap = np.reshape(phi,(-1,self.num_energies()))/Z
        return gap
    
    """Method to find Tc if there is Cooper log. Only valid for weak coupling."""
    def find_tc_ir(self,om_cut = 1.0,temp_list = None,save_data = False):
       
        if temp_list is None:
            temp_list = np.array([0.05,0.08,0.1,0.12])

        else:
            #check if temp_list is a list of positive numbers
            if not isinstance(temp_list,(list,np.ndarray)):
                raise TypeError('temp_list must be a list!')
            if all(isinstance(x,float) and x > 0 for x in temp_list) == False:
                raise TypeError('weight_list must be a list of only positive floats!')

            

        temp_list = np.array(temp_list)

        L_list = np.log(1/temp_list)

        lambda_list = []

        for temp in temp_list:
            print('\ntemp:',temp)
            print('num_freq:',self.num_freq(temp))
            print('size of basis: {}\n'.format(self.num_freq(temp)*self.num_energies()))
            lambda_bar = self.find_lambda_bar(temp,om_cut)
            lambda_list.append(lambda_bar)

        #save data points
        if save_data == True:

            np.savetxt('lambda_data.txt',np.transpose(np.array([L_list,lambda_list])))

        #fit line to data points
        [m,b]=np.polyfit(L_list,lambda_list,1)
        
        x = np.linspace(np.min(L_list)/2.0,np.max(L_list)*1.5)
        #plt.plot(x,m*x+b)
        
        #if m is positive, T_c exists. otherwise, no.
        if m > 0:
            print('tc is',np.exp(-1*(1-b)/m))
            return np.exp(-1*(1-b)/m)
        else:
            print('tc does not exist')
            return -1

    """
    SOLVE NONLINEAR ELIASHBERG EQUATIONS FOR Z AND PHI AT A GIVEN TEMP
    """
        

    def find_Z_phi_nonlin(self,temp, verbose = 0):

        max_iter = 100
        tol = 0.001
        
        npts = self.vec_size(temp)
        
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        
        #initialize Z and phi
        Z_in = np.ones((num_mf,num_en))
        Z_out= np.empty((num_mf,num_en))

        phi_in = np.random.rand(num_mf,num_en)
        phi_out= np.empty((num_mf,num_en))

        if verbose == 1:
            print('Size of Z:',npts)
       
        #define arrays which will be used to iterate Z and phi
        mf_list = self.mat_freq_list(temp)
        interaction_array = self.pot_fn(mf_list[:,None,None,None],self.en_list[None,:,None,None],mf_list[None,None,:,None],self.en_list[None,None,None,:])



        for idx in range(max_iter):

            #iterate Z, phi
            Z_out[:,:] = 1.0 - temp/mf_list[:,None] * np.einsum('ijkl,kl',interaction_array,\
                    self.weight_list[None,:]*self.dos(self.en_list[None,:]) * mf_list[:,None] * Z_in \
                    /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu,2) + np.power(phi_in,2)))
            
            phi_out[:,:] = -1.0 * temp * np.einsum('ijkl,kl',interaction_array,\
                    self.weight_list[None,:]*self.dos(self.en_list[None,:]) * phi_in \
                    /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu,2) + np.power(phi_in,2)))
    
            diff = np.linalg.norm(Z_out - Z_in) + np.linalg.norm(phi_out - phi_in)
            
            if verbose == 1:
                print('\nIteration:',idx)
                print('diff:',diff)

            if  diff < tol:
                if verbose == 1:
                    print('\nConvergence achieved')
                    print('diff:',diff)
                return (Z_out, phi_out)
           
           #initialize new Z_in and phi_in from old Z_out and phi_out for next iteration
            Z_in[:,:] = Z_out
            phi_in[:,:] = phi_out
        
        #if we have reached this point, convergence has not been achieved
        if verbose == 1:
            print('Convergence has not been achieved')
            print('diff: {}'.format(diff))
        return (Z_out,phi_out)

    def find_Z_phi_nonlin_fft(self,temp, verbose = 0):

        max_iter = 100
        tol = 0.0001
        
        npts = self.vec_size(temp)
        
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        
        #initialize Z and phi
        Z_in = np.ones((num_mf,num_en))
        Z_out= np.empty((num_mf,num_en))

        phi_in = np.random.rand(num_mf,num_en)
        phi_out= np.empty((num_mf,num_en))

        if verbose == 1:
            print('Size of Z:',npts)
        
        
        #define function that expands list from just bosonic frequencies
        def expand_boson(boson_freq_vec):
            #make longer list that includes space for fermionic frequencies
            all_freq_vec = np.zeros(2*len(boson_freq_vec) - 1)
            all_freq_vec[::2] = boson_freq_vec

            #reorder frequencies to standard order for fourier transform
            all_freq_vec = np.roll(all_freq_vec,1-len(boson_freq_vec))
            return all_freq_vec


        #define function that expands list from just fermionic frequencies
        def expand_fermion(fermion_freq_vec):
            #make longer list that includes space for bosonic frequencies
            all_freq_vec = np.zeros(2*len(fermion_freq_vec) + 1)
            all_freq_vec[1::2] = fermion_freq_vec
            #reorder frequencies to standard order for fourier transform
            all_freq_vec = np.roll(all_freq_vec,-1*len(fermion_freq_vec))
            return all_freq_vec

        #define function that picks out fermionic frequencies
        def contract_fermion(all_freq_vec):
            #reorder frequencies to standard order before fourier transform(most negative frequency first)
            #select out only the fermionic frequencies (the rest should be zero)
            fermion_freq_vec = np.roll(all_freq_vec, int((len(all_freq_vec)-1)/2))[1::2]
            return fermion_freq_vec

        #evaluate pot_fn at bosonic frequencies, store for iteration below
        mf_list = self.mat_freq_list(temp)
        shifted_mf_list = mf_list - np.pi * temp
        boson_freq_list = np.append(shifted_mf_list, shifted_mf_list[-1] + 2*np.pi*temp)
        pot_freq_arr = self.pot_fn(boson_freq_list[:,None,None], self.en_list[None,:,None],0.0, self.en_list[None,None,:])

        #use the above list to fourier transform to (imaginary) time
        pot_freq_arr = np.apply_along_axis(expand_boson,0,pot_freq_arr)
        pot_time_arr = fft.fft(pot_freq_arr,axis=0)

 
        #define arrays which will be used to iterate Z and phi
        mf_list = self.mat_freq_list(temp)
        fermion_arr = np.empty(np.shape(Z_in))
        expanded_fermion_arr = np.empty((2*len(mf_list)+1,len(self.en_list)))


        for idx in range(max_iter):

            ####iterate Z
            fermion_arr[:,:] = self.weight_list[None,:]*self.dos(self.en_list[None,:]) * mf_list[:,None] * Z_in \
                         /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu,2) + np.power(phi_in,2))

            #expand zeroth axis to all frequencies (not just fermionic)
            expanded_fermion_arr[:,:] = np.apply_along_axis(expand_fermion,0,fermion_arr)
            
            #fourier transform expanded array to (imaginary) time
            expanded_fermion_arr[:,:] = fft.fft(expanded_fermion_arr,axis=0)

            #contract energies and multiply arrays along time axis, then extract fermionic frequencies
            fermion_arr[:,:] = np.apply_along_axis(contract_fermion,0,fft.ifft(np.einsum('ijk,ik->ij',pot_time_arr,expanded_fermion_arr),axis=0))

            Z_out[:,:] = 1.0 - temp/mf_list[:,None] * fermion_arr 

            ####iterate phi 
            fermion_arr[:,:] = self.weight_list[None,:]*self.dos(self.en_list[None,:]) * phi_in \
                         /(np.power(mf_list[:,None]*Z_in,2) + np.power(self.en_list[None,:]-self.mu,2) + np.power(phi_in,2))

            #expand zeroth axis to all frequencies (not just fermionic)
            expanded_fermion_arr[:,:] = np.apply_along_axis(expand_fermion,0,fermion_arr)

            #fourier transform expanded array to (imaginary) time
            expanded_fermion_arr[:,:] = fft.fft(expanded_fermion_arr,axis=0)
            
            #contract energies and multiply arrays along time axis, then extract fermionic frequencies
            phi_out[:,:] = -1.0 * temp * np.apply_along_axis(contract_fermion,0,fft.ifft(np.einsum('ijk,ik->ij',pot_time_arr,expanded_fermion_arr),axis=0))


    
            diff = np.linalg.norm(Z_out - Z_in) + np.linalg.norm(phi_out - phi_in)
            
            if verbose == 1:
                print('\nIteration:',idx)
                print('diff:',diff)

            if  diff < tol:
                if verbose == 1:
                    print('\nConvergence achieved')
                    print('diff:',diff)
                return (Z_out, phi_out)
           
           #initialize new Z_in and phi_in from old Z_out and phi_out for next iteration
            Z_in[:,:] = Z_out
            phi_in[:,:] = phi_out
        
        #if we have reached this point, convergence has not been achieved
        if verbose == 1:
            print('Convergence has not been achieved')
            print('diff: {}'.format(diff))
        return (Z_out,phi_out)

def phonon_int(coupling):
    def pot_fn(mf_1,en_1,mf_2,en_2):
        return 0.1-1.0*coupling/(1.0 + np.power(mf_1-mf_2,2))
    return pot_fn





if __name__ == "__main__":
    ph_int = phonon_int(0.35)
    
    pot_fn = ph_int
    freq_cut = 5
    mu = 5.0
    en_list = np.linspace(0,10,100)
    weight_list = 0.1 * np.ones(100)
    dos = lambda x : 1.0
    
    eberg = Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut, 
                  mu = mu, 
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)
    
    
    
    temp = 0.02
    mf_list = eberg.mat_freq_list(temp)
    en_list = eberg.en_list
    
    num_mf = len(mf_list)
    num_en = len(en_list)
    
    gap = eberg.find_gap_ir(temp).reshape(num_mf,num_en)
    
    plt.scatter(mf_list,gap[:,0])
    plt.show()
