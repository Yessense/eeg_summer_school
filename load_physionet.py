import mne
from matplotlib import pyplot as plt

dir_path = 'C:/Users/Fedosov/Downloads/S004R03.edf'


raw = mne.io.read_raw_edf(dir_path)

raw.plot_psd()
plt.show()






