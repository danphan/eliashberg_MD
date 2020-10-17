import numpy as np
import matplotlib.pyplot as plt
import eberg as eb
a = 0
b = 1
num_en = 100
temp = 0.024698732722294503
freq_cut = 5.0

en_list,weight_list = np.polynomial.legendre.leggauss(num_en)
en_list = (b-a)/2.0 * en_list + (a+b)/2.0
weight_list = weight_list * (b-a)/2.0

mf_list = eb.mf_list(temp,freq_cut) 

gap = np.loadtxt('gap_file.txt',skiprows=2)
#plt.plot(en_list,gap[int(len(mf_list)/2),:],'o')
plt.plot(mf_list,gap[:,0],'o')
plt.show()
