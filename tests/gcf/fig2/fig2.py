"""
REPRODUCE FIG 2 OF PAPER
"""
import matplotlib.pyplot as plt
import numpy as np
#import eberg as eb


def int_freq(mf_1, mf_2):
    return -1.0/(1.0+np.power(mf_1-mf_2,2))

def int_mom(rho,mu):
    kappa_squared = np.sqrt(mu*rho)*4.0/np.pi 
    def interaction(en_1,en_2):
        return np.sqrt(rho/(en_1*en_2))/(2*np.pi) * np.log((np.power(np.sqrt(en_1)+np.sqrt(en_2),2)+kappa_squared)/(np.power(np.sqrt(en_1)-np.sqrt(en_2),2)+kappa_squared))
    return interaction

def dos(x):
    return np.sqrt(x)

phonon_freq = 0.1
rydberg = 13.6

rho = rydberg/phonon_freq
en_list = np.linspace(0.00001,40,600)



plt.figure(figsize=(10,10))

#make first subplot
plt.subplot(3,1,1)
mu = 10

coupling_list = np.array([dos(en_list)*int_mom(rho,mu)(en,en_list) for en in en_list])
for i in range(30):
    plt.plot(en_list-mu,coupling_list[i,:])
plt.xlim(-5,5)

#make second subplot
plt.subplot(3,1,2)
mu = 1

coupling_list = np.array([dos(en_list)*int_mom(rho,mu)(en,en_list) for en in en_list])
for i in range(30):
    plt.plot(en_list-mu,coupling_list[i,:],'o')
plt.xlim(-5,5)



#make third subplot
plt.subplot(3,1,3)
mu = .1

coupling_list = np.array([dos(en_list)*int_mom(rho,mu)(en,en_list) for en in en_list])
for i in range(30):
    plt.plot(en_list-mu,coupling_list[i,:])
plt.xlim(-5,5)



plt.show()



