"""
Create plot of Tc vs phonon frequency, for different values of phonon frequency omegaL, setting mu = 0
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb

###########WRITE FILE NAME WHICH WILL STORE MU AND TC_LIST######## 
filename = 'phonon_freq_tc_file.txt'

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

def dos(x):
    return np.sqrt(x)


rydberg = 13.6
mu = 0.0000000000001
freq_cut = 13.0
num_en = 150

#make en_list,weight_list
en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
a=0.
b=20.
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0


phonon_freq_list = [0.00001, 0.0001,0.001,0.01,0.1,1.]
tc_list = []

#create file which will contain omega_L and tc's
with open(filename,'w') as f:
    f.writelines('omega_L Tc\n')


for phonon_freq in phonon_freq_list:
    rho = rydberg/phonon_freq
    pot_fn = tot_interaction(rho,mu)

    eberg = eb.Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut,
                  mu = mu,
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)

    tc_guess =1 
    tc = eberg.find_tc_eigval(tc_guess)
    print('\ntc:',tc)
    tc_list.append(tc)
    with open(filename,'a') as f:
        f.writelines('{} {}\n'.format(phonon_freq,tc))

plt.loglog(phonon_freq_list,tc_list,'o')
plt.ylabel('Tc')
plt.xlabel('phonon freq')
plt.savefig('fig6b.pdf')
plt.show()






