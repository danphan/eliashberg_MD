import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, LinearOperator
from useful_functions import solve_linear

def matsubara(idx,temp):
    return (2*idx+1)*np.pi*temp

def mf_list(temp,freq_cut):
    npts = 2*int(freq_cut/(2*np.pi*temp))

    if num_mf > 2:
        return matsubara(np.arange(npts)-npts/2,temp)
    else:
        raise Exception('number of matsubara frequencies is less than 2. increase freq_cut.')

class Eberg:
    def __init__(self,pot_fn,freq_cut, en_list, weight_list, mu, dos):
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
            raise Exception('number of matsubara frequencies is less than 2. increase freq_cut.')

    def num_energies(self):
        return len(self.en_list)
        
    def vec_size(self,temp):
        return self.num_freq(temp) * self.num_energies()
   
    #does the same thing as mf_list function defined above
    def mat_freq_list(self,temp):
        npts = self.num_freq(temp)
        return matsubara(np.arange(npts)-npts/2,temp)
    
    def __iter_Z(self,temp,Z_in):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
        
        Z_in = Z_in.reshape((num_mf, num_en))
        
        mf_list = self.mat_freq_list(temp)
        
        Z_out = np.empty(npts)
        
        #reshape lists for easy broadcasting
        w_list_shaped = self.weight_list[None,:]
        en_list_shaped = self.en_list[None,:]
        dos_list_shaped = self.dos(en_list_shaped)
        mf_list_shaped = mf_list[:,None]
        
        for idx in range(npts):
            mf_idx, en_idx = np.divmod(idx,num_en)
            mf = mf_list[mf_idx]
            en = self.en_list[en_idx]
            
            Z_out[idx] = 1.0 - temp/mf * np.sum(w_list_shaped * dos_list_shaped \
                                                * self.pot_fn(mf,en,mf_list_shaped,en_list_shaped)\
                                                * mf_list_shaped * Z_in\
                                                / (np.power(mf_list_shaped * Z_in,2) + np.power(en_list_shaped-self.mu,2)))
        return Z_out
    
    def find_Z(self,temp,verbose = 0, self_consistent = True):
        max_iter = 100
        tol = 0.001
        
        npts = self.vec_size(temp)
        
        Z_in = np.ones(npts)
        Z_out= np.ones(npts)
        if verbose == 1:
            print('Size of Z:',npts)
        
        for idx in range(max_iter):
            Z_out[:] = self.__iter_Z(temp,Z_in)
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
            
            Z_in[:] = Z_out
        
        #if we have reached this point, convergence has not been achieved
        if verbose == 1:
            print('Convergence has not been achieved')
        return Z_out
    
    def __find_kernel(self,temp,verbose = 0):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
        
        if verbose == 1:
            print('\nKernel Size:',npts,'x',npts)
        
        Z = self.find_Z(temp,verbose)
        
        Z = Z.reshape((num_mf,num_en))
        
        mf_list = self.mat_freq_list(temp)
        
        
        #reshape lists for easy broadcasting
        w_list_shaped = self.weight_list[None,:]
        en_list_shaped = self.en_list[None,:]
        dos_list_shaped = self.dos(en_list_shaped)
        mf_list_shaped = mf_list[:,None]
        
        kernel = np.empty((npts,npts))
        
        for idx in range(npts):
            mf_idx, en_idx = np.divmod(idx,num_en)
            mf = mf_list[mf_idx]
            en = self.en_list[en_idx]
            kernel_array = -1.0 * temp * self.pot_fn(mf, en, mf_list_shaped, en_list_shaped) \
            * w_list_shaped * dos_list_shaped \
            / (np.power(mf_list_shaped * Z,2) + np.power(en_list_shaped - self.mu,2))
            kernel[idx,:] = kernel_array.reshape(npts)
            
        if verbose == 1:
            print('Kernel found')
            
        return kernel
    

    def __find_kernel_sorted(self,temp,om_cut = 1.0, verbose = 0):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
        
        if verbose == 1:
            print('\nKernel Size:',npts,'x',npts)
        
        Z = self.find_Z(temp,verbose)
                
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

        #use above lists to construct sorted kernel
        kernel = np.empty((npts,npts))
        for idx in range(npts):
            kernel[idx,:] = -1.0 * temp \
            * self.pot_fn(mf_list_sorted[idx],en_list_sorted[idx],mf_list_sorted,en_list_sorted)\
            * weight_list_sorted * dos_list_sorted\
            / (np.power(Z_sorted * mf_list_sorted,2) + np.power(en_list_sorted - self.mu,2))
    
        return kernel
    
    """ONLY FOR IMPLICIT RENORMALIZATION. APPROPRIATE ONLY WHEN THERE IS A COOPER LOG"""
    def find_lambda_bar(self,temp,om_cut,return_gap = False):
        num_en = self.num_energies()
        num_mf = self.num_freq(temp)
        npts = self.vec_size(temp)
                
        mf_list = self.mat_freq_list(temp)
        
#         #make long list for en_tot
#         mf_idx_list,en_idx_list = np.divmod(np.arange(npts),num_en)
#         mf_list_long = mf_list[mf_idx_list]
#         en_list_long = self.en_list[en_idx_list]
#         en_tot_list = np.sqrt(np.power(mf_list_long,2) + np.power(en_list_long-self.mu,2))
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
            raise Exception('Choose a smaller om_cut! Om_cut must be smaller than the energy cutoffs to ensure a separation between low and high energies!')


                
        #use this idx_bd to break up kernel
        kernel = -1.0*self.__find_kernel_sorted(temp)
        
        K11 = kernel[:idx_bd, :idx_bd]
        K12 = kernel[:idx_bd, idx_bd:]
        K21 = kernel[idx_bd:, :idx_bd]
        K22 = kernel[idx_bd:, idx_bd:]
        
        #use these to find lambda_bar
        phi = np.ones(npts)
        
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
            
            return phi_unsorted
            
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
    
    def __find_eigval(self,temp):
        kernel = self.__find_kernel(temp)
        return np.real(eigs(kernel,k=1,which='LR')[0])[0]
    
    def find_tc_eigval(self,tc0,tol = 0.0001,max_iter = 10):
        temp_list = []
        eigval_list = []
        
        temp_1 = tc0
        eigval_1 = self.__find_eigval(temp_1)
        bool_1 = eigval_1 > 1
        
        print('temp:',temp_1)
        print('eigval:',eigval_1)
        
        temp_2 = tc0/2.
        eigval_2 = self.__find_eigval(temp_2)
        bool_2 = eigval_2 > 1
        
        print('temp:',temp_2)
        print('eigval:',eigval_2)
        
        temp_list.append(temp_1)
        temp_list.append(temp_2)
        eigval_list.append(eigval_1)
        eigval_list.append(eigval_2)
        
        
        
        for num in range(max_iter):
            if bool_2 == bool_1:
                
                #reassign temp_2 to temp_1
                #reassign eigval_2 to eigval_1
                temp_1 = temp_2
                eigval_1 = eigval_2
                
                temp_2 *= 0.5
                eigval_2 = self.__find_eigval(temp_2)
                bool_2 = eigval_2 > 1
                
                print('temp:',temp_2)
                print('eigval:',eigval_2)
                
                temp_list.append(temp_2)
                eigval_list.append(eigval_2)
    
            else:
                print("Signs are different! We have a range between T_c's")
                break
                
        if bool_1 == bool_2:
            print("Newton's method not applied. Have not bounded Tc between 2 values")
            print("Upper bound on T_c:",temp_2)
            return temp_2
        
        m = (eigval_2-eigval_1)/(temp_2-temp_1)
        temp = temp_1 + (1-eigval_1)/m
        eigval = self.__find_eigval(temp)
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
                temp_1 = temp
                eigval_1 = eigval
                bool_1 = sign
                
                
            m = (eigval_2 - eigval_1)/(temp_2-temp_1)
            temp = temp_1 + (1-eigval_1)/m
            eigval = self.__find_eigval(temp)
            sign  = eigval > 1
            
            print('temp:',temp)
            print('eigval:',eigval)
            
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
        
    def find_gap(self):
        print('\nFINDING GAP\n')
        tc = self.find_tc_eigval(0.1)
        print('\ntc:',tc)
        e_val,e_vec = eigs(self.__find_kernel(tc),k=1,which='LR')
        phi = np.real(e_vec)
        return np.reshape(phi,len(phi))/self.find_Z(tc)
    
    """Method to find Tc if there is Cooper log. Only valid for weak coupling."""
    def find_tc_ir(self,om_cut = 1.0,temp_list = None,save_data = False):
       
        if temp_list is None:
            temp_list = np.array([0.05,0.08,0.1,0.12])

        temp_list = np.array(temp_list)


        L_list = np.log(1/temp_list)

        lambda_list = []

        for temp in temp_list:
            print('\ntemp:',temp)
            print('num_freq:',self.num_freq(temp))
            print('size of basis: {}\n'.format(self.num_freq(temp)*self.num_energies()))
            lambda_bar = self.find_lambda_bar(temp,om_cut)
            lambda_list.append(lambda_bar)

        #p = plt.scatter(L_list,lambda_list)

#         plt.savefig('figure__'+str(mu)+'.png',format='png')

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
