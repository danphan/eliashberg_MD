"""
Make plot of Tc vs mu 
"""
import matplotlib.pyplot as plt
import numpy as np
import eberg as eb

###########WRITE FILE NAME WHICH WILL STORE MU AND TC_LIST######## 
filename = 'mu_tc_file.txt'

def int_freq(mf_1, mf_2):
    return 0.-1.0/(1.0+np.power(mf_1-mf_2,2))

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
num_en_1 = 70
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

#####SAVE EN_LIST TO TEXT FILE ############# 


mu_list = [0.1]
tc_list = []

##create file which will contain mu and tc's
#with open(filename,'w') as f:
#    f.writelines('mu tc\n')


for mu in mu_list:
    pot_fn = tot_interaction(rho,mu)

    eberg = eb.Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut,
                  mu = mu,
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)

#    tc_guess = 0.25
#    temp_list = [0.03,0.04,.05]
#    tc = eberg.find_tc_eigval(tc0 = 0.02)
#    print('\ntc:',tc)

#    gap = eberg.find_gap(tc0=0.1)
##    tc_list.append(tc)
##    with open(filename,'a') as f:
##        f.writelines('{} {}\n'.format(mu,tc))

    temp = 0.025
    gap_ir = eberg.find_gap_ir(temp)
    with open('gap_file_{}.txt'.format(mu),'w') as ff:
        ff.writelines('mu temp freq_cut num_energies a b\n')
        ff.writelines('{} {} {} {} {} {}\n'.format(mu,temp,freq_cut,len(en_list),en_list[0],en_list[-1]))
        np.savetxt(ff,gap_ir)
    
    #save en_list
    np.savetxt('en_list_{}.txt'.format(mu),en_list)


##plt.semilogx(kappa_squared_list,tc_list,'o')
##plt.ylabel('tc')
##plt.xlabel('kappa squared')
##plt.savefig('fig6.pdf')
##plt.show()
##
##
#
#
#
#
