import os
import pathlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import pandas as pd
import librosa
import librosa.display as ld
from IPython.display import Audio
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
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
    3 = Comer
    4 = Ir_para_o_quarto
    5 = Subir_cama_sofa
    6 = Pegar_no_colo

    * [occurrenceID] = um identificador numérico para distinguir diferentes ocorrências do som dentro da gravação original;
    * [sliceID] = um identificador numérico para distinguir diferentes fatias tiradas da mesma ocorrência;

* start: a hora de início da fatia na gravação original de Freesound
* end: a hora de término da fatia na gravação original de Freesound
* salience: uma classificação de saliência (subjetiva) do som. 1 = primeiro plano, 2 = plano de fundo.
* fold: o número da pasta (1-10) ao qual este arquivo foi alocado.
* class: descrição da classe
"""
metadata = pd.read_csv('metadata/SoundsPipinho.csv')
#print(metadata)

fsID = []
classID = []
occurID = []
sliceID = []
full_path = []

for root, dirs, files in tqdm(os.walk('audios/')):
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

sound_list = [ 'Choro','Abrir_a_porta', 'Chegada_do_dono', 'Comer', 'Ir_para_o_quarto','Subir_cama_sofa', 'Pegar_no_colo']
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

#plt.figure(figsize=(18,7))
#sns.countplot(x=df['classID'])
#plt.show()

#print(df.shape)

n_files = df.shape[0]
rnd = np.random.randint(0,n_files)
fname = df['path'][rnd]
#print(fname)

data,sample_rate = librosa.load(fname,sr=None) #22050 Hz
print('Canais: ',len(data.shape))
print('Número total de amostras: ', data.shape[0])
print('Arquivos: ', fname)
print('Taxa de amostragem: ',sample_rate)
print('Duração: ',len(data)/sample_rate)

info = df.iloc[rnd].values
#print(info)
title_txt = 'Som: ' + info [1]

def features_extractor(file_name):
  data, sample_rate = librosa.load(file_name, sr = None, res_type = 'kaiser_fast')
  mfccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc=40)
  mfcss_features_scaled = np.mean(mfccs_features.T, axis = 0)
  return mfcss_features_scaled

extracted_features = []
for path in tqdm(df['path'].values):
  #print(path)
  data = features_extractor(path)
  extracted_features.append([data])

extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature'])
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(df['classID'].tolist())

extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature'])
feature = np.array(extracted_features_df['feature'].tolist())
classID = np.array(df['classID'].tolist())

def train_test(feature,classID,tax):
  feature_inicio=[]
  classID_inicio=[]
  for i in range(len(feature)):
    linha=[]
    for j in range(len(feature[i])):
      linha.append(feature[i][j])
    feature_inicio.append(linha)
    classID_inicio.append(classID[i])
  #print(feature_inicio)
  #print(classID_inicio)
  feature_train=[]
  classID_train=[]
  feature_test=[]
  classID_test=[]
  feature_val=[]
  classID_val=[]
  qtyClass=[0,0,0,0,0,0,0]
  for i in range(len(classID)):
    qtyClass[sound_list.index(classID[i])]+=1
  #print(qtyClass)
  for i in range(len(qtyClass)):
    n=int(qtyClass[i]*tax)
    if n<2:
      n=2
    k=0
    #Alimentando Teste
    for j in range(len(classID)):
      if classID_inicio[j]==sound_list[i]:
        feature_test.append(feature_inicio[j])
        feature_inicio.remove(feature_inicio[j])
        classID_test.append(classID_inicio[j])
        classID_inicio.remove(classID_inicio[j])
        k+=1
        if k==int(n/2):
          break
    k=0
    #Alimentando Validação
    for j in range(len(classID)):
      if classID_inicio[j]==sound_list[i]:
        feature_val.append(feature_inicio[j])
        feature_inicio.remove(feature_inicio[j])
        classID_val.append(classID_inicio[j])
        classID_inicio.remove(classID_inicio[j])
        k+=1
        if k==int(n/2):
          break
    k=0
  for j in range (len(classID_inicio)):
    feature_train.append(feature_inicio[j])
    classID_train.append(classID_inicio[j])
  
  feature_train=np.array(feature_train)
  classID_train=np.array(classID_train)
  feature_test=np.array(feature_test)
  classID_test=np.array(classID_test)
  feature_val=np.array(feature_val)
  classID_val=np.array(classID_val)

  labelencoder = LabelEncoder()
  classID_train = to_categorical(labelencoder.fit_transform(classID_train))
  classID_test = to_categorical(labelencoder.fit_transform(classID_test))
  classID_val = to_categorical(labelencoder.fit_transform(classID_val))
 
  return  feature_train,classID_train,feature_test,classID_test,feature_val,classID_val,labelencoder

"""
X_train, y_train, X_test, y_test, X_val, y_val,labelencoder = train_test(feature,classID,0.2)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)

X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]
"""

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state=1)

X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]

def get_model():
  model = Sequential()

  model.add(Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1], 1)))
  model.add(Dropout(0.4))
  model.add(MaxPooling1D(pool_size=(4)))

  model.add(Conv1D(128, 10, padding='same',))
  model.add(Activation('relu'))
  model.add(Dropout(0.4))
  model.add(MaxPooling1D(pool_size=(4)))

  model.add(Flatten())

  model.add(Dense(units = 64))
  model.add(Dropout(0.4))
  model.add(Dense(units = 7))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
  #model.summary()
  return model

def treinamento(num_epochs,num_batch_size):
  model = get_model()
  model.load_weights("saved_models/barking_classification.h5")
  #Treinamento da base
  num_epochs = num_epochs
  num_batch_size = num_batch_size

  checkpointer = ModelCheckpoint(filepath = 'saved_models/barking_classification_checkpoint.hdf5',
                                verbose = 1, save_best_only = True)
  start = datetime.now()
  history = model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs,
                      validation_data = (X_val, y_val), callbacks = [checkpointer], verbose = 1)
  duration = datetime.now() - start
  print('Duração do treinamento: ', duration)
  print('')

  model.save_weights("saved_models/barking_classification.h5")

  #Avaliação do modelo
  #Treinamento
  print('Treinamento')
  score = model.evaluate(X_train, y_train)
  print(score)
  print('')

  #Validação
  print('Validação')
  score = model.evaluate(X_val, y_val)
  print(score)
  print('')

  #Testes
  print('Teste')
  score = model.evaluate(X_test, y_test)
  print(score)
  print('')

  #Grafico de validação
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation']);
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation']);
  plt.show()


  predictions = model.predict(X_test)
  predictions = predictions.argmax(axis = 1)
  predictions = predictions.astype(int).flatten()
  predictions = (labelencoder.inverse_transform((predictions)))
  predictions = pd.DataFrame({'Classes previstas': predictions})
  #print(predictions)
  #print('')

  actual = y_test.argmax(axis = 1)
  actual = actual.astype(int).flatten()
  actual = labelencoder.inverse_transform(actual)
  actual = pd.DataFrame({'Classes reais': actual})
  #print(actual)
  #print('')

  final_df = actual.join(predictions)
  #print(final_df)
  #print('')

  cm = confusion_matrix(actual, predictions)
  cm = pd.DataFrame(cm, index = [i for i in labelencoder.classes_], columns = [i for i in labelencoder.classes_])
  #print(cm)

  plt.figure(figsize = (12,10))
  ax = sns.heatmap(cm, linecolor = 'white', cmap = 'Greys_r', linewidth=1, annot = True, fmt = '')
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  plt.title('Matriz de confusão', size = 20)
  plt.xlabel('Classes previstas', size = 14)
  plt.ylabel('Classes reais', size = 14)
  plt.show()

  print(classification_report(actual, predictions))

def get_info(data, sample_rate):
  print('Canais: ', len(data.shape))
  print('Número total de amostras: ', data.shape[0])
  print('Taxa de amostragem: ', sample_rate)
  print('Duração: ', len(data) / sample_rate)

def predict_sound(arquivo_audio,info = False, plot_waveform = False, plot_spectrogram = False):
  audio, sample_rate = librosa.load(arquivo_audio, sr = None, res_type = 'kaiser_fast')
  mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc=40)
  mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
  mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
  mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]
  
  prediction = model.predict(mfccs_scaled_features)
  prediction = prediction.argmax(axis=1)
  prediction = prediction.astype(int).flatten()
  prediction = labelencoder.inverse_transform((prediction))

  print('Classificação/resultado: ', prediction)

  if info:
    get_info(audio, sample_rate)
  
  """  
  if plot_waveform:
    plt.figure(figsize=(14,5))
    plt.title('Tipo de som: ' + str(prediction[0].upper()), size = 16)
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    ld.waveplot(audio, sr=sample_rate)
    plt.show()

  #Espectrogramas de MFCCs
  if plot_spectrogram:
    plt.figure(figsize=(14,5))
    mfccs_db = librosa.amplitude_to_db(np.abs(mfccs_features))
    plt.title('Tipo de som: ' + str(prediction[0].upper()), size = 16)
    ld.specshow(mfccs_db, x_axis='time', y_axis='log', cmap='Spectral')
    plt.colorbar(format='%+2.f dB')
    plt.show()
  """

def teste(arquivo_audio):
  audio, sample_rate = librosa.load(arquivo_audio, sr = None, res_type = 'kaiser_fast')
  predict_sound(arquivo_audio, info = True, plot_waveform=True, plot_spectrogram=True)

#treinamento(100,35)
model = get_model()
model.load_weights("saved_models/barking_classification.h5")
model.summary

teste(r'audios\Folder2\655907-2-0-5.wav')
teste(r'audios\Folder2\655147-3-0-0.wav')
teste(r'audios\Folder2\655166-4-0-3.wav')
teste(r'audios2\Train\0-Choro\654246-0-0-4.wav')
teste(r'audios2\Test\5-Subir_cama_sofa\655195-5-0-0.wav')
teste(r'audios\Folder1\655164-6-0-1.wav')
  
"""
0 = Choro
1 = Abrir_a_porta
2 = Chegada_do_dono
3 = Comer
4 = Ir_para_o_quarto
5 = Subir_cama_sofa
6 = Pegar_no_colo
"""