"""
Solve for Z and chi using linear Eliashberg equation for a given temperature 
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb

###########WRITE FILE NAME WHICH WILL STORE MU AND TC_LIST######## 
filename = 'mu_tc_file.txt'

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
freq_cut = 10.0
num_en_1 = 100
num_en_2 = 70

#make en_list,weight_list
en_list_1,weight_list_1 = np.polynomial.legendre.leggauss(num_en_1)
a=0.
b=.1
en_list_1 = (b-a)/2.0 * en_list_1 + (a+b)/2.0
weight_list_1 = weight_list_1 * (b-a)/2.0

en_list_2,weight_list_2 = np.polynomial.legendre.leggauss(num_en_2)
c=100.
en_list_2 = (c-b)/2.0 * en_list_2 + (b+c)/2.0
weight_list_2 = weight_list_2 * (c-b)/2.0

en_list = np.concatenate((en_list_1,en_list_2))
weight_list = np.concatenate((weight_list_1,weight_list_2))

mu = 0.01


pot_fn = tot_interaction(rho,mu)

eberg = eb.Eberg(pot_fn = pot_fn,
              freq_cut = freq_cut,
              mu = mu,
              en_list = en_list,
              weight_list = weight_list,
              dos = dos)


temp = 0.04
Z,chi = eberg.find_Z_chi(temp,verbose=1)

mf_list = eb.mf_list(temp,freq_cut)

ax1 = plt.subplot(1,2,1)
ax1.plot(en_list,chi[int(len(mf_list)/2),:],'o')
ax1.set_title('chi vs energy')
ax2 = plt.subplot(1,2,2)
ax2.plot(mf_list,Z[:,0],'o')
ax2.set_title('Z vs frequency')
plt.savefig('Z_phi_nonlinear_temp_{}_mu_{}.pdf'.format(temp,mu))
plt.show()
