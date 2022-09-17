import numpy as np
import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14,6)
from IPython.display import HTML, Audio
import librosa
import librosa.display as ld
import seaborn as sn
sn.set()

librosa.util.list_examples()

filename = librosa.util.example_audio_file()
#filename = librosa.ex('vibeace')
#filename = librosa.util.example_audio_file()
filename = librosa.ex('trumpet')
mono_data, sample_rate = librosa.load(filename, duration = 60)