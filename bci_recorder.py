import os
import sys
import logging
import numpy as np
import scipy.signal as sg
from datetime import datetime
import keyboard


from pylsl import StreamInlet, resolve_byprop
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error, StreamOutlet
import xml.etree.ElementTree as ET
from lsl_inlet import LSLInlet

import csv

session_num = 0
num_exp = 5



# record data
def record_data(exp_settings, inlet, nseconds):
    fs = int(inlet.get_frequency())
    channels = inlet.get_channels_labels()
    n_channels = len(channels)
    logging.info('Stream channels = {}'.format(np.array(channels)))

    # prepare recording utils
   

    buffer = np.empty((nseconds * fs + 100 * fs, n_channels))

    n_samples_received = 0
   
    
    while(not keyboard.is_pressed('s')):
        pass
        # сбрасываем буфер
        #chunk,t_stamp = inlet.get_next_chunk()
            
    while(n_samples_received/fs <= nseconds):

            chunk, t_stamp = inlet.get_next_chunk()

            if chunk is not None:
                n_samples_in_chunk = len(chunk)
                buffer[n_samples_received:n_samples_received+n_samples_in_chunk,
                          :n_channels] = chunk
                n_samples_received += n_samples_in_chunk

    # save recorded data
    recorded_data = buffer[:int(round(nseconds*fs))]
    return recorded_data, channels, fs

if __name__ == '__main__':

    # create results folder
    #timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    results_path = 'results/session_{}/{}_{}/'.format(session_num,'bci_exp_nostim',  num_exp)
    os.makedirs(results_path)

    # setup logger
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(results_path, 'log.txt'))]
    logging.basicConfig(format='%(asctime)s> %(message)s', level=logging.INFO, handlers=handlers)

    # main parameters
    PROBES_DURATION_1 = 3
    STATES_CODES_DICT = {'Rest': 0, 'Left': 1, 'Right':3 , 'Legs': 2}
    EXP_SETTINGS_PROBES = {
        'exp_name': 'example',
        'lsl_stream_name': 'pseudo_data',
       

        'max_chunklen': 10
    }
  
    # connect to LSL stream
    inlet = LSLInlet(EXP_SETTINGS_PROBES['lsl_stream_name'])



    # record probes
    recorded_data, channels, fs = record_data(EXP_SETTINGS_PROBES, inlet, nseconds = 10)

    time = np.arange(recorded_data.shape[0])/fs

    with open(results_path+'/data.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['time_ms']+channels)
        for i in range(recorded_data.shape[0]):
            csvwriter.writerow([time[i]] + recorded_data[i,:].tolist())

 