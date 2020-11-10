"""
new potential, (beyond Thomas-Fermi)
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


def int_mom_beyond_tf(rho,mu):
    #keep in mind that mu is in terms of omega_L, the phonon frequency
    kappa_squared = np.sqrt(mu*rho)*4.0/np.pi
    
    def g(k1,k2,costheta):
        return 1.0
        x = np.sqrt(1.0/mu)*np.sqrt(np.power(k1,2) + np.power(k2,2) - 2 * k1 * k2 * costheta)
        return 0.5 - 0.5/x * (1 - 0.25 * np.power(x,2)) * np.log(np.abs((1-x/2.0)/(1+x/2.0)))

    num_points = 50

    def interaction(en_1,en_2):
        k1 = np.atleast_1d(np.sqrt(en_1))
        k2 = np.atleast_1d(np.sqrt(en_2))

        cos_list, weights = np.polynomial.legendre.leggauss(num_points)

        #change to match k1 and k2


        #change shape of k1 and k2 to add extra axis for cos data
        denominator_list = np.power(k1,2) + np.power(k2,2) - 2.0 * k1 * k2* cos_list + kappa_squared * g(k1,k2,cos_list)
        return np.sqrt(rho) / np.pi * np.sum(weights/denominator_list)

    return interaction

def int_mom_beyond_tf_vectorized(rho,mu):
    #keep in mind that mu is in terms of omega_L, the phonon frequency
    kappa_squared = np.sqrt(mu*rho)*4.0/np.pi
    
    def g(k1,k2,costheta):
        return 1.0
        x = np.sqrt(1.0/mu)*np.sqrt(np.power(k1,2) + np.power(k2,2) - 2 * k1 * k2 * costheta)
        return 0.5 - 0.5/x * (1 - 0.25 * np.power(x,2)) * np.log(np.abs((1-x/2.0)/(1+x/2.0)))

    num_points = 200

    def interaction(en_1,en_2):
        k1 = np.atleast_1d(np.sqrt(en_1))
        k2 = np.atleast_1d(np.sqrt(en_2))

        k1,k2 = np.broadcast_arrays(k1,k2)



        cos_list, weights = np.polynomial.legendre.leggauss(num_points)

        #change to match k1 and k2
        num_dimensions = len(np.shape(k1))

        cos_list = np.expand_dims(cos_list,axis = tuple(range(num_dimensions)))
        weights = np.expand_dims(weights,axis = tuple(range(num_dimensions)))

        #change shape of k1 and k2 to add extra axis for cos data
        k1 = k1[...,None]
        k2 = k2[...,None]
        denominator_list = np.power(k1,2) + np.power(k2,2) - 2.0 * k1 * k2* cos_list + kappa_squared * g(k1,k2,cos_list)
        return np.sqrt(rho) / np.pi * np.sum(weights/denominator_list,axis=-1)

    return interaction

def tot_interaction_beyond_tf(rho,mu):
    def interaction(mf_1,en_1,mf_2,en_2):
#        #vectorize int_mom_beyond_tf
#        int_mom_vec = np.vectorize(int_mom_beyond_tf(rho,mu))

        return int_freq(mf_1,mf_2)*int_mom_beyond_tf_vectorized(rho,mu)(en_1,en_2)
    return interaction
#rho = rydberg/phonon_freq
#mu = 0.001
#
#en_list = np.linspace(0.00001,3.0,200)
#old_pot_fn = int_mom(rho,mu)
#new_pot_fn = int_mom_beyond_tf(rho,mu)
#
#old_pot_list = old_pot_fn(0.1,en_list)
#new_pot_list = new_pot_fn(0.1,en_list)
#
#plt.plot(en_list,old_pot_list,'o',label='thomas fermi')
#plt.plot(en_list,new_pot_list + 0.001,'o',label='beyond thomas fermi')
#
#plt.legend()
#plt.show()

phonon_freq = 0.1
rydberg = 13.6

rho = rydberg/phonon_freq
freq_cut = 5.0
num_en_1 = 40
num_en_2 = 40

#make en_list,weight_list
en_list_1,weight_list_1 = np.polynomial.legendre.leggauss(num_en_1)
a=0.
b=0.1
en_list_1 = (b-a)/2.0 * en_list_1 + (a+b)/2.0
weight_list_1 = weight_list_1 * (b-a)/2.0

en_list_2,weight_list_2 = np.polynomial.legendre.leggauss(num_en_2)
c=100.
en_list_2 = (c-b)/2.0 * en_list_2 + (b+c)/2.0
weight_list_2 = weight_list_2 * (c-b)/2.0

en_list = np.concatenate((en_list_1,en_list_2))
weight_list = np.concatenate((weight_list_1,weight_list_2))


mu_list = [0.1]
tc_list = []

##create file which will contain mu and tc's
#with open(filename,'w') as f:
#    f.writelines('mu tc\n')


for mu in mu_list:
    pot_fn = tot_interaction_beyond_tf(rho,mu)

    eberg = eb.Eberg(pot_fn = pot_fn,
                  freq_cut = freq_cut,
                  mu = mu,
                  en_list = en_list,
                  weight_list = weight_list,
                  dos = dos)

#    tc_guess = 0.25
#    temp_list = [0.03,0.04,.05]
    tc = eberg.find_tc_ir()
#    print('\ntc:',tc)

#    gap = eberg.find_gap(tc0=0.1)
##    tc_list.append(tc)
##    with open(filename,'a') as f:
##        f.writelines('{} {}\n'.format(mu,tc))

    gap_ir = eberg.find_gap_ir(temp=tc+0.01)
    with open('gap_file_{}.txt'.format(mu),'w') as ff:
        ff.writelines('mu temp freq_cut num_energies a b\n')
        ff.writelines('{} {} {} {} {} {}\n'.format(mu,tc+0.01,freq_cut,len(en_list),en_list[0],en_list[-1]))
        np.savetxt(ff,gap_ir)

    #save en_list
    np.savetxt('en_list_{}.txt'.format(mu),en_list)

