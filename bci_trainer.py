import os
import sys
import logging
import numpy as np
import scipy.signal as sg
from datetime import datetime

from pylsl import StreamInlet, resolve_byprop
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error, StreamOutlet
import xml.etree.ElementTree as ET
from lsl_inlet import LSLInlet
import PyQt6.QtWidgets as QtWidgets
from visualizer import Panel
import csv

session_num = 0
num_exp = 3



# record data
def record_data(exp_settings, inlet, panel):
    fs = int(inlet.get_frequency())
    channels = inlet.get_channels_labels()
    n_channels = len(channels)
    logging.info('Stream channels = {}'.format(np.array(channels)))

    # prepare recording utils
    block_durations = [exp_settings['blocks'][block_name]['duration']
                       for block_name in exp_settings['sequence']]
    n_seconds = sum(block_durations)
    n_blocks = len(exp_settings['sequence'])
    buffer = np.empty((n_seconds * fs + 100 * fs, n_channels + 1))

    STIM_CHANNEL = n_channels

    n_samples_received = 0
    residual_samples = 0
    for j_block, block_name in enumerate(exp_settings['sequence']):
        current_block = exp_settings['blocks'][block_name]
        if 'message' in current_block:
            panel.ChangeText(current_block['message'])
            QtWidgets.QApplication.processEvents()
        n_samples = fs * current_block['duration']
        block_id = current_block['id']
        logging.info('Block #{}/{} id{} "{}" ({}s)'.format(
            j_block, n_blocks, block_id, block_name, current_block['duration']))
        n_samples_received_in_block = 0
        if residual_samples != 0:
            buffer[n_samples_received:n_samples_received+residual_samples,
            :n_channels] = chunk[-residual_samples:, :]
            buffer[n_samples_received:n_samples_received+residual_samples,
            STIM_CHANNEL] = block_id
            n_samples_received_in_block += residual_samples
            n_samples_received += residual_samples

        while n_samples_received_in_block < n_samples:
            chunk, t_stamp = inlet.get_next_chunk()

            if chunk is not None:
                n_samples_in_chunk = len(chunk)

                if n_samples_in_chunk > n_samples-n_samples_received_in_block:
                    residual_samples = n_samples_in_chunk - (n_samples-n_samples_received_in_block)
                    upper_idx = n_samples_received + n_samples_in_chunk - residual_samples
                    print(n_samples_in_chunk)
                else:
                    upper_idx = n_samples_received + n_samples_in_chunk
                    residual_samples = 0

                if residual_samples>0:
                    buffer[n_samples_received:upper_idx,
                    :n_channels] = chunk[:-residual_samples,:]

                else:
                    buffer[n_samples_received:upper_idx,
                    :n_channels] = chunk

                buffer[n_samples_received:upper_idx,
                STIM_CHANNEL] = block_id

                n_samples_received_in_block += n_samples_in_chunk-residual_samples
                n_samples_received += n_samples_in_chunk-residual_samples

    # save recorded data
    recorded_data = buffer[:n_samples_received]
    return recorded_data, channels, fs

if __name__ == '__main__':

    # create results folder
    #timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    results_path = 'results/session_{}/{}_{}/'.format(session_num,'bci_exp',  num_exp)
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
        'blocks': {
            'Rest': {'duration': 5, 'id': STATES_CODES_DICT['Rest'], 'message': '+'},
            'Legs': {'duration': PROBES_DURATION_1, 'id': STATES_CODES_DICT['Legs'], 'message': 'Ноги'},
            'Right': {'duration': PROBES_DURATION_1, 'id': STATES_CODES_DICT['Right'], 'message': 'Правая рука'},
            'Left': {'duration': PROBES_DURATION_1, 'id': STATES_CODES_DICT['Left'], 'message': 'Левая рука'},
            'Prepare': {'duration': 10, 'id': 42, 'message': 'Готовность 10 секунд'}
        },
        'sequence': ['Legs', 'Left', 'Right']*2,
        'max_chunklen': 10
    }
    np.random.shuffle(EXP_SETTINGS_PROBES['sequence'])
    EXP_SETTINGS_PROBES['sequence'] = ['Prepare']+EXP_SETTINGS_PROBES['sequence']

    # connect to LSL stream
    inlet = LSLInlet(EXP_SETTINGS_PROBES['lsl_stream_name'])

    # visualizer
    app = QtWidgets.QApplication([])
    text_panel = Panel()
    text_panel.InitWindow()
    text_panel.update()
    # QtWidgets.QApplication.processEvents()
    #app.exec()

    # record probes
    recorded_data, channels, fs = record_data(EXP_SETTINGS_PROBES, inlet, text_panel)

    time = np.arange(recorded_data.shape[0])/fs

    with open(results_path+'/data.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['time_ms']+channels+['state'])
        for i in range(recorded_data.shape[0]):
            csvwriter.writerow([time[i]] + recorded_data[i,:].tolist())

 