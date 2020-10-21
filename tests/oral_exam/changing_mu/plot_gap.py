import numpy as np
import matplotlib.pyplot as plt
import eberg as eb
import re
import os


#loop through files in current directory and find the files containing the gap
for entry in os.scandir('.'):
    if entry.is_file() and entry.name.startswith('gap_file'):

        #for each file, import and plot the gap
        with open(entry.name,'r') as f:
            for i, line in enumerate(f):
                if i == 1:
                    string_nums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                    mu, temp, freq_cut, num_energies, a, b = [float(x) for x in string_nums]
                    gap = np.loadtxt('gap_file_{}.txt'.format(mu),skiprows=2)
                    en_list = np.loadtxt('en_list_{}.txt'.format(mu))
                    mf_list = eb.mf_list(temp,freq_cut)
                    plt.loglog(en_list,np.abs(gap[int(len(mf_list)/2),:])/np.max(np.abs(gap)),'o',label= 'mu = {}'.format(mu))

#show plot
plt.legend()
plt.title('gap vs energy')
plt.savefig('gap_vs_energy_loglog.pdf')
plt.show()

