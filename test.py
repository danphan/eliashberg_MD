"""
CREATE NEW POTENTIAL THAT NEEDS TO BE INTEGRATED OVER
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb


def int_freq(mf_1, mf_2):
    return -1.0/(1.0+np.power(mf_1-mf_2,2))

def int_mom(rho,mu):
    kappa_squared = 4.0/np.pi * np.sqrt(mu*rho)
    def interaction(en_1,en_2):
        return np.sqrt(rho/(en_1*en_2))/(2*np.pi) * np.log((np.power(np.sqrt(en_1)+np.sqrt(en_2),2)+kappa_squared)/(np.power(np.sqrt(en_1)-np.sqrt(en_2),2)+kappa_squared))
    return interaction


def tot_interaction(rho,mu):
    def interaction(mf_1,en_1,mf_2,en_2):
        return int_freq(mf_1,mf_2)*int_mom(rho,mu)(en_1,en_2)
    return interaction

phonon_freq = 0.1
rydberg = 13.6

rho = rydberg/phonon_freq

mu = 1

pot_fn = tot_interaction(rho,mu)

def dos(x):
    return np.sqrt(x)


freq_cut = 50.0
num_en = 200

#make en_list,weight_list
en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
a=0.
b=100.
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0


eberg = eb.Eberg(pot_fn = pot_fn,
              freq_cut = freq_cut,
              mu = mu,
              en_list = en_list,
              weight_list = weight_list,
              dos = dos)


temp = 0.2
Z = eberg.find_Z(temp)
Z = np.reshape(Z,(-1,num_en))

mf_list = eb.mf_list(temp,freq_cut)

plt.semilogy(mf_list[-int(len(mf_list)/2):], Z[:,0][-int(len(mf_list)/2):]-1,'bo')
plt.show()


#tc_guess = 0.06
#tc = eberg.find_tc_eigval(tc_guess)
#print('\ntc:',tc)
