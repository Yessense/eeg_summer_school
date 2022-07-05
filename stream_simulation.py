from pylsl.pylsl import StreamOutlet, StreamInfo
import numpy as np
import time


chNames = ['C3','C4','Cz']
chCount = len(chNames)

class BCIStreamOutlet(StreamOutlet):
    def __init__(self, name = 'pseudo_data', fs = 500):
        # create stream info
        info = StreamInfo(name=name, type='BCI', channel_count=chCount, nominal_srate=fs,
                          channel_format='float32', source_id='myuid34234')

        # set channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
        chns = info.desc().append_child("channels")
        for chname in chNames:
            ch = chns.append_child("channel")
            ch.append_child_value("label", chname)

        # init stream
        super(BCIStreamOutlet, self).__init__(info,chunk_size = 14)


fs = 500
outlet = BCIStreamOutlet(fs = fs)

start_time = time.time()
model_time = 0.0
Ns = 14
while(1):
    cur_time = time.time()
    if cur_time-start_time > model_time+Ns/fs:

        x = np.random.randn(chCount*Ns,).tolist()
        outlet.push_chunk(x)
        model_time += Ns/fs

    if model_time > 500.0:
        del outlet
        break