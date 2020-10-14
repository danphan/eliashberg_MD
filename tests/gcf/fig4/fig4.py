"""
REPRODUCE FIGURE 4
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

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
freq_cut = 30.0
num_en = 150


#make en_list,weight_list
en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
a=0.
b=100.
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0

mu_list = [10,1,0.1,0.01]

plt.figure(figsize=(10,10))
###########################
for i in range(4):
    mu = mu_list[i]
    pot_fn = tot_interaction(rho,mu)
    
    en_idx,nearest_val = find_nearest(en_list,mu)
    print('nearest val',nearest_val)
    
    
    eberg = eb.Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut,
                  mu = mu,
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)
    
    temp = 0.2
    Z_nsc = eberg.find_Z(temp,self_consistent= False)
    Z_nsc = np.reshape(Z_nsc,(-1,num_en))


    mf_list = eb.mf_list(temp,freq_cut)

    #MAKE Z_MIGDAL_ELIASHBERG, WHERE COUPLING IS CONSTANT
    coupling = dos(mu) * int_mom(rho,mu)(mu,mu)
    Z_me = np.empty(len(mf_list))
    for idx in range(len(Z_me)):
        mf = mf_list[idx]
        zeta_list = np.arctan((b-mu)/mf_list) + np.arctan(mu/mf_list)
        Z_me[idx] = 1 - coupling * temp /mf * np.sum(int_freq(mf_list,mf)*zeta_list)

    plt.subplot(2,2,i+1)
    plt.loglog(mf_list[-int(len(mf_list)/2):],Z_me[-int(len(mf_list)/2):]-1,'o',label='migdal eliashberg')
    plt.loglog(mf_list[-int(len(mf_list)/2):],Z_nsc[-int(len(mf_list)/2):,en_idx]-1,'o',label='keeping energy dependence')
    
plt.legend()
plt.show()
