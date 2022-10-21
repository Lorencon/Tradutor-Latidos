import os
import pathlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display as ld
from IPython.display import Audio
from tqdm import tqdm
import tensorflow
#print('Versão Tensorflow: ', tensorflow.__version__)
#print('Versão librosa: ', librosa.__version__)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

"""
Metadados
Cada arquivo de áudio é composto por:
* slice_file_name: 
O nome do arquivo de áudio. 
  * O nome tem o seguinte formato: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, onde:
    * [fsID] = é o ID Freesound da gravação da qual este trecho (fatia) foi retirado. 
    Podemos testar isto copiando o ID de um destes sons de nossa base de dados e pesquisar no site www.freesound.org.
    * [classID] = um identificador numérico da classe de som (0-7);
    0 = Choro
    1 = Abrir_a_porta
    2 = Chegada_do_dono
    4 = Comer
    5 = Ir_para_o_quarto
    6 = Passear
    7 = Pegar_no_colo
    8 = Subir_cama_sofá

    * [occurrenceID] = um identificador numérico para distinguir diferentes ocorrências do som dentro da gravação original;
    * [sliceID] = um identificador numérico para distinguir diferentes fatias tiradas da mesma ocorrência;

* start: a hora de início da fatia na gravação original de Freesound
* end: a hora de término da fatia na gravação original de Freesound
* salience: uma classificação de saliência (subjetiva) do som. 1 = primeiro plano, 2 = plano de fundo.
* fold: o número da pasta (1-10) ao qual este arquivo foi alocado.
* class: descrição da classe
"""
metadata = pd.read_csv('C:/Users/Ramom Landim/Desktop/TG/Tradutor-Latidos/metadata/SoundsPipinho.csv')
metadata

fsID = []
classID = []
occurID = []
sliceID = []
full_path = []

for root, dirs, files in tqdm(os.walk('C:/Users/Ramom Landim/Desktop/TG/Tradutor-Latidos/audios')):
  #print(root)
  #print(dirs)
  #print(files)
  for file in files:
    try:
      #print(file.split('-'))
      fs = int(file.split('-')[0])
      class_ = int(file.split('-')[1])
      occur = int(file.split('-')[2])
      slice_ = file.split('-')[3]
      slice_ = int(slice_.split('.')[0])
      #print(fs)

      fsID.append(fs)
      classID.append(class_)
      occurID.append(occur)
      sliceID.append(slice_)
    
      full_path.append((root, file))
    except ValueError:
      continue

sound_list = [ 'Choro','Abrir_a_porta', 'Chegada_do_dono', 'Comer', 'Ir_para_o_quarto', 'Passear','Pegar_no_colo', 'Subir_cama_sofá']
sound_dict = {em[0]:em[1] for em in enumerate(sound_list)}
#print(sound_dict)

df = pd.DataFrame([fsID, classID, occurID, sliceID, full_path]).T
#print(df)

df.columns = ['fsID', 'classID', 'occurID', 'sliceID', 'path']
#print(df)

df['classID'] = df['classID'].map(sound_dict)
df['path'] = df['path'].apply(lambda x: x[0] + '/' + x[1])
#print(df)

#print(df.describe())
#print(df['classID'].value_counts())

plt.figure(figsize=(18,7))
sns.countplot(df['classID'])
plt.show()