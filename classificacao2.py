import seaborn as sns
sns.set()
import itertools
import glob
import random
import os
import csv
import librosa
import librosa.display as ld
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from tensorflow.python.keras import optimizers
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
import tensorflow_io as tfio

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])
  return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

metadata = pd.read_csv('metadata/SoundsPipinho.csv')
sound_list = ['Choro','Abrir_a_porta','Chegada_do_dono','Comer','Ir_para_o_quarto','Subir_cama_sofa','Pegar_no_colo']

def get_random_file():
    test_list = glob.glob('audios2/Test/*/*.wav')
    random_audio_path = random.choice(test_list)
    #print(random_audio_path)
    return random_audio_path

def get_dog_data(audio_path, plot=True):
    data, sample_rate = librosa.load(audio_path, sr = 16000)
    classCode = audio_path.split('-')[-3]
    #print('Classe: ',sound_list[int(classCode)])
    return data

random_audio = get_random_file()
scores, embeddings, spectrogram = yamnet_model(get_dog_data(random_audio))
class_scores = tf.reduce_mean(scores, axis = 0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]
#print('Som: ', infered_class)
#print ('shape: ', embeddings.shape)

spec = audio_classifier.YamNetSpec(keep_yamnet_and_custom_heads=True,frame_step=3*audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,frame_length = 6*audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

train_data = audio_classifier.DataLoader.from_folder(spec,'audios2/Train',cache=True)
train_data, validation_data = train_data.split(0.8)
test_date = audio_classifier.DataLoader.from_folder(spec,'audios2/Test',cache=True)

batch_size = 64
epochs = 100  

model = audio_classifier.create(train_data, spec, validation_data, batch_size=batch_size, epochs=epochs)