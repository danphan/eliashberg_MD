"""
Make plot of Tc vs mu 
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb

###########WRITE FILE NAME WHICH WILL STORE MU AND TC_LIST######## 
filename = 'mu_tc_file.txt'

def int_freq(mf_1, mf_2):
    return 1.-1.0/(1.0+np.power(mf_1-mf_2,2))

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
freq_cut = 5.0
num_en = 200

#make en_list,weight_list
en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
a=0.
b=100.
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0

mu_list = [0.0001]
tc_list = []

#create file which will contain mu and tc's
with open(filename,'w') as f:
    f.writelines('mu tc\n')


for mu in mu_list:
    pot_fn = tot_interaction(rho,mu)

    eberg = eb.Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut,
                  mu = mu,
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)

#    tc_guess = 0.25
    temp_list = [0.04,.05]
    tc = eberg.find_tc_ir(temp_list = temp_list,save_data = True)
    print('\ntc:',tc)
    tc_list.append(tc)
    with open(filename,'a') as f:
        f.writelines('{} {}\n'.format(mu,tc))
#
#plt.semilogx(kappa_squared_list,tc_list,'o')
#plt.ylabel('tc')
#plt.xlabel('kappa squared')
#plt.savefig('fig6.pdf')
#plt.show()
#
#




