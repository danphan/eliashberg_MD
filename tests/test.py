"""
CREATE NEW POTENTIAL THAT NEEDS TO BE INTEGRATED OVER
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb


def int_freq(mf_1, mf_2):
    return -1.0/(1.0+np.power(mf_1-mf_2,2))

def int_mom(rho,mu):
    kappa_squared = np.sqrt(mu*rho)*4.0/np.pi 
    def interaction(en_1,en_2):
        return np.sqrt(rho/(en_1*en_2))/(2*np.pi) * np.log((np.power(np.sqrt(en_1)+np.sqrt(en_2),2)+kappa_squared)/(np.power(np.sqrt(en_1)-np.sqrt(en_2),2)+kappa_squared))
    return interaction


def tot_interaction(rho,mu):
    def interaction(mf_1,en_1,mf_2,en_2):
        return int_freq(mf_1,mf_2)*int_mom(rho,mu)(en_1,en_2)
    return interaction

def mu_from_kappa_squared(rho,kappa_squared):
    return np.power(np.pi*kappa_squared/4.0,2)/rho

def dos(x):
    return np.sqrt(x)

phonon_freq = 0.1
rydberg = 13.6

rho = rydberg/phonon_freq
mu = 1
freq_cut = 30.0
num_en = 150

pot_fn = tot_interaction(rho,mu)

#make en_list,weight_list
en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
a=0.
b=100.
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0


#coupling_list = [dos(mu) * int_mom(rho,mu)(mu,mu)for mu in en_list] 
#
#def analytical_coupling(mu):
#    return np.sqrt(rho/mu)*np.log(1+np.pi*np.sqrt(mu/rho))/(2*np.pi)
#
#analytical_coupling_list = analytical_coupling(en_list)
#
#plt.plot(en_list,analytical_coupling_list,label='analytical')
#plt.plot(en_list,coupling_list,label='my interaction')
#plt.legend()
#plt.show()



eberg = eb.Eberg(pot_fn = pot_fn,
              freq_cut = freq_cut,
              mu = mu,
              en_list = en_list,
              weight_list = weight_list,
              dos = dos)



#for en_idx in range(num_en_plot):
#    plt.loglog(mf_list[-int(len(mf_list)/2):], Z[:,en_idx][-int(len(mf_list)/2):]-1,'o')
#plt.show()

#tc_guess = 0.055
#tc = eberg.find_tc_eigval(tc_guess)
#print('\ntc:',tc)


#temp = tc
temp = 0.2
#Z_nsc = eberg.find_Z(temp,self_consistent= False)
#Z_nsc = np.reshape(Z_nsc,(-1,num_en))

Z_sc = eberg.find_Z(temp,self_consistent= True)
Z_sc = np.reshape(Z_sc,(-1,num_en))
#
mf_list = eb.mf_list(temp,freq_cut)
#
num_en_plot = 30
plt.semilogx(mf_list[-int(len(mf_list)/2):],Z_sc[-int(len(mf_list)/2):,0],'o')
plt.show()
##write Z_nsc analytic, from eq. 27 of Maria, Note that this disregards all energy dependence of the interaction and density of states!!!
##TYPO in equatino 24 of maria, should have an extra omega_n' in the sum
#coupling = dos(mu) * int_mom(rho,mu)(mu,mu)
##print('coupling',coupling)
##Z_nsc_analytic = 1 - coupling * temp/mf_list * np.matmul(int_freq(mf_list[:,None],mf_list[None,:]),(np.arctan((b-mu)/mf_list) + np.arctan(mu/mf_list))/mf_list)
#Z_nsc_analytic = np.empty(len(mf_list))
#
#for i in range(len(mf_list)):
#    mf = mf_list[i]
#    Z_nsc_analytic[i] = 1 - coupling * temp/mf * np.sum(int_freq(mf,mf_list)*(np.arctan((b - mu)/mf_list) + np.arctan(mu/mf_list)))

#plt.loglog(mf_list[-int(len(mf_list)/2):],Z_nsc_analytic[-int(len(mf_list)/2):]-1,'o')
#plt.loglog(mf_list[-int(len(mf_list)/2):],Z_nsc[-int(len(mf_list)/2):,10]-1,'o')
#plt.show()

#ratio = (Z_nsc_analytic)[:,None]/(Z_sc)

#ratio = np.empty(np.shape(Z_sc))
#
#for idx in range(len(en_list)):
#    ratio[:,idx] = (Z_nsc_analytic-1)/(Z_sc[:,idx]-1)
##
#
#print(en_list[num_en_plot-1])

#####print Z_sc and Z_nsc
#for en_idx in range(num_en_plot):
#    plt.loglog(mf_list[-int(len(mf_list)/2):], Z_sc[-int(len(mf_list)/2):,en_idx]-1,'o')
#plt.show()

#ratio = (Z_nsc-1)/(Z_sc - 1)

#for en_idx in range(num_en_plot):
#    plt.semilogx(mf_list[-int(len(mf_list)/2):], ratio[-int(len(mf_list)/2):,en_idx],'o')
##plt.ylim(0.9,1.6)
#plt.show()
#




