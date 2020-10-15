import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('lambda_data.txt')
log_list,lambda_list = np.transpose(data)

plt.plot(log_list,lambda_list,'o')
plt.show()
