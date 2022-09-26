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
import tensorflow as tf
sn.set()

librosa.util.list_examples()

# Formato OGG: https://pt.wikipedia.org/wiki/Ogg
filename = librosa.util.example_audio_file()
#filename = librosa.ex('trumpet')
mono_data, sample_rate = librosa.load(filename, duration = 60)

#Audio Mono - mais rapido, porem com menos qualidade
mono_data, sample_rate = librosa.load(filename, duration = 60)
print('Vetor: ', mono_data)
print('Canais: ', mono_data.shape)
print('Número total de amostras: ',mono_data.shape[0]) #tamanho total do vetor (Numero de amostragems)
print('Taxa de amostragem: ',sample_rate)
print('Duração do áudio: ', len(mono_data)/sample_rate)
print('Duração do áudio: ', librosa.get_duration(mono_data))
print('')


#Audio(data = mono_data, rate = sample_rate) #Escuta do audio no Google colab

#Audio stereo - Mais demorado, porem mais qualidade
stereo_data, sample_rate = librosa.load(filename, mono=False ,duration = 60)
print('Vetor: ', stereo_data)
print('Canais: ', stereo_data.shape)
print('Número total de amostras: ',stereo_data.shape[0]) #tamanho total do vetor (Numero de amostragems)
print('Taxa de amostragem: ',sample_rate)
print('Duração do áudio: ', librosa.get_duration(stereo_data))
print('')

#Configuração da taxa de amostragem
data,sample_rate = librosa.load(filename,duration = 60, sr = 8000)
print ('Taxa de amostragem: ', sample_rate)
print('Amostras: ', data)
print('Quantidade de amostras: ', len(data))

#Reamostragem

filename = librosa.util.example_audio_file()
start = time.time()
data , sample_rate = librosa.load(filename,duration=60,res_type='kaiser_best')
best = time.time()-start

start = time.time()
data , sample_rate = librosa.load(filename,duration=60,res_type='kaiser_fast')
fast = time.time()-start

start = time.time()
data , sample_rate = librosa.load(filename,duration=60,res_type='scipy')
scipy = time.time()-start

start = time.time()
data , sample_rate = librosa.load(filename,duration=60,res_type='polyphase')
poly = time.time()-start

print('Tempo de carregamento por tipo de reamostragem')

print('kaiser_best',best)
print('kaiser_fast',fast)
print('scipy',scipy)
print('polyphase',poly)

#Separação de um som harmonico do percusivo
filename = librosa.ex('nutcracker')
y, sr = librosa.load(filename)
y_harmonic, y_percussive = librosa.effects.hpss(y)

print('Taxa de amostragem: ', sr)
print('Quantidade de amostras: ', len(y))
print('Duração: ', librosa.get_duration(y))
print('Canais: ', y.shape)

# Detecção de início e sintetização de click
y, sr = librosa.load(librosa.ex('trumpet'))
# https://librosa.org/doc/main/generated/librosa.onset.onset_strength.html
onset_env = librosa.onset.onset_strength(y = y, sr = sr, max_size = 5)
onset_env.shape, type(onset_env)

onset_frames = librosa.onset.onset_detect(onset_envelope = onset_env, sr = sr)

times = librosa.times_like(onset_env, sr = sr)
times.shape, type(times)

plt.plot(times, onset_env, label = 'Onset strength')
plt.vlines(times[onset_frames], 0, onset_env.max(), color = 'r', linestyle = '--', label = 'Onsets')
plt.legend()
plt.show()

onset_times = librosa.onset.onset_detect(onset_envelope = onset_env, sr = sr, units = 'time')
y_clicks = librosa.clicks(times = onset_times, length = len(y), sr = sr)
Audio(data = y + y_clicks, rate = sr,autoplay=True)