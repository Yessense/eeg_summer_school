

import csv
import numpy as np
from scipy.io import loadmat


fs = 1200

pathdir = 'C:/Users/Fedosov/Downloads/matlab_probes_train.mat'


arr = loadmat(pathdir)
data = arr['data']
del arr


state = data[:,-1].astype(int)
data = data[:,0]
time = np.arange(data.shape[0])/fs

with open('C:/Users/Fedosov/Downloads/train_data_1.csv', 'w',  newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['time_ms', 'data'])
    for i in range(data.shape[0]):
      csvwriter.writerow([time[i], data[i]])
      
      

with open('C:/Users/Fedosov/Downloads/train_states_1.csv', 'w',  newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['time_ms', 'state'])
    for i in range(data.shape[0]):
      csvwriter.writerow([time[i], state[i]])



