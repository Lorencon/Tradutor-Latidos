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

"""filename = librosa.util.example_audio_file()
#filename = librosa.ex('trumpet')

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
print('Quantidade de amostras: ', len(data)) """

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
