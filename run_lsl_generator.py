from pylsl.pylsl import StreamOutlet, StreamInfo
import numpy as np
import time
import pandas as pd



class BCIStreamOutlet(StreamOutlet):
     def __init__(self, chCount, chNames, fs, Ns, name = 'real_time'):
         # create stream info
         info = StreamInfo(name=name, type='BCI', channel_count=chCount, nominal_srate=fs,
                           channel_format='float32', source_id='myuid34234')

         # set channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
         chns = info.desc().append_child("channels")
         for chname in chNames:
             ch = chns.append_child("channel")
             ch.append_child_value("label", chname)

         # init stream
         super(BCIStreamOutlet, self).__init__(info,chunk_size = Ns)

def run_lsl_generator():
      
   
     
      session_num = 0
      num = 1
      path_dir = 'results/session_{}/'.format(session_num)
  
      
    
 
      dataframe = pd.read_csv(path_dir+'bci_exp_nostim'+str(num)+'/data.csv')
      data = (dataframe.to_numpy()[:,1:])
    
         
          
          
      
     
      fs = 500
      chNames = list()
      for i in range(19):
          chNames.append(str(i))
      chCount = len(chNames)
      Ns = 10 # number of samples to send

     


     
      outlet = BCIStreamOutlet(chCount, chNames, fs)

      start_time = time.time()
      model_time = 0.0
      
      samples_count = 0
      while(1):
          cur_time = time.time()
          if cur_time-start_time > model_time+Ns/fs:

              x = np.random.randn(chCount*Ns,).tolist()
              outlet.push_chunk(x)
              model_time += Ns/fs
              samples_count += Ns
              
          if samples_count >= data.shape[0]:
              del outlet
              break