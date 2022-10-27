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

#plt.title(title_txt.upper(),size = 16)
#librosa.display.waveplot(data,sr=sample_rate)
#plt.show()
#Audio(data=data,rate=sample_rate) apenas no Jupyter


#Apenas no Jupyter
random_sample = df.groupby('classID').sample(1)
audio_samples, labels = random_sample['path'].tolist(),random_sample['classID'].tolist()
rows = 5
cols = 2

"""
fig, axs = plt.subplot(rows,cols,figsize=(15,15))
index = 0
for col in range (cols):
  for row in range(rows):
    data,sample_rate = librosa.load(audio_samples[index],sr=None)
    librosa.display.waveplot(data,sample_rate,ax=axs[row][col])
    axs[row][col].set_title('{}'.format(labels[index]))
    index+=1
    plt.show()
fig.tight_layout()

#Espectrogramas de STFT
fig, axs = plt.subplots(rows, cols, figsize=(20,20))
index = 0
for col in range(cols):
  for row in range(rows):
    data, sample_rate = librosa.load(audio_samples[index], sr = None)
    stft = librosa.stft(y = data)
    stft_db = librosa.amplitude_to_db(np.abs(stft))
    img = librosa.display.specshow(stft_db, x_axis = 'time', y_axis = 'log', ax = axs[row][col], cmap = 'Spectral')
    axs[row][col].set_title('{}'.format(labels[index]))
    fig.colorbar(img, ax = axs[row][col], format='%+2.f dB')
    index += 1
fig.tight_layout()

#Espectrogramas de MFCCs
"""
fig, axs = plt.subplots(rows, cols, figsize=(20,20))
index = 0
for col in range(cols):
    for row in range(rows):
        if(index!=7):
          data, sample_rate = librosa.load(audio_samples[index], sr = None)
          mfccs = librosa.feature.mfcc(y = data, sr=sample_rate, n_mfcc=40)
          mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
          img = librosa.display.specshow(mfccs_db, x_axis="time", y_axis='log', ax=axs[row][col], cmap = 'Spectral')
          axs[row][col].set_title('{}'.format(labels[index]))
          fig.colorbar(img, ax=axs[row][col], format='%+2.f dB')
          index += 1
fig.tight_layout()
plt.show()


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

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state=1)

X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]

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

#Treinamento da base
num_epochs = 80
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath = 'saved_models/ambient_sound_classification.hdf5',
                               verbose = 1, save_best_only = True)
start = datetime.now()
history = model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs,
                    validation_data = (X_val, y_val), callbacks = [checkpointer], verbose = 1)
duration = datetime.now() - start
print('Duração do treinamento: ', duration)
print('')

#Avaliação do modelo
#Treinamento
score = model.evaluate(X_train, y_train)
print(score)
print('')

#Validação
score = model.evaluate(X_val, y_val)
print(score)
print('')

#Testes
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
print(predictions)
print('')

actual = y_test.argmax(axis = 1)
actual = actual.astype(int).flatten()
actual = labelencoder.inverse_transform(actual)
actual = pd.DataFrame({'Classes reais': actual})
print(actual)
print('')

final_df = actual.join(predictions)
print(final_df)
print('')

cm = confusion_matrix(actual, predictions)
#cm = pd.DataFrame(cm, index = [i for i in labelencoder.classes_], columns = [i for i in labelencoder.classes_])
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

def predict_sound(arquivo_audio, info = False, plot_waveform = False, plot_spectrogram = False):
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

  if plot_waveform:
    plt.figure(figsize=(14,5))
    plt.title('Tipo de som: ' + str(prediction[0].upper()), size = 16)
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    ld.waveplot(audio, sr=sample_rate)
    plt.show()

  if plot_spectrogram:
    plt.figure(figsize=(14,5))
    mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
    plt.title('Tipo de som: ' + str(prediction[0].upper()), size = 16)
    ld.specshow(mfccs_db, x_axis='time', y_axis='log', cmap='Spectral')
    plt.colorbar(format='%+2.f dB')
    plt.show()

audio, sample_rate = librosa.load('audios/Folder1/654159-2-0-1.wav', sr = None, res_type = 'kaiser_fast')
predict_sound('audios/Folder1/654159-2-0-1.wav', info = True, plot_waveform=True, plot_spectrogram=True)

  
