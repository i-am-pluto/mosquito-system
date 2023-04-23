import pickle
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import lite as tflite
import time
import serial
import h5py
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution1D, MaxPooling2D, Convolution2D, DepthwiseConv2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from tensorflow.keras.activations import relu, softmax
import time
from keras.callbacks import ModelCheckpoint
from keras.callbacks import RemoteMonitor
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score

import pyaudio


wav_root = '../data/audio_1sec/'

labels = pd.read_csv('../labels/audio_1sec.csv',  header=None,
                     names=["path", "yes", "no", "not_sure", "subject_set"])
labels['path'] = labels['path'].astype(str) + '.wav'

labels['res'] = (labels['yes'].astype(int)*1 + labels['not_sure'].astype(int)
                 * 0.5) / (labels['yes'] + labels['no'] + labels['not_sure']) >= 0.5
n = len(labels.index.values.tolist())

y = np.zeros((n, 2))

y[:, 1] = np.array(np.array(labels['res']).astype(int)).astype(int)
y[:, 0] = 1-y[:, 1].astype(int)

feature_type = 'mfcc'


# save_name = 'not_sure_single_into_0_5'
# hf = h5py.File('../proc_data/data_' + feature_type + save_name + '.h5', 'r')
# spec_matrix_read = np.array(
#     hf.get('../proc_data/data_' + feature_type + '_majority_labels_' + save_name))
# hf.close()
# hf = h5py.File('../proc_data/label_' + feature_type + save_name + '.h5', 'r')
# y = np.array(hf.get('../proc_data/label_' + feature_type +
#              '_majority_labels_' + save_name))
# hf.close()

# spec_matrix_db = np.zeros_like(spec_matrix_read)
# for i, spec in enumerate(spec_matrix_read):
#     spec_matrix_db[i] = librosa.power_to_db(spec, ref=np.max)


# def get_model():
#     X_train, X_test, y_train, y_test = train_test_split(
#         spec_matrix_db, y, test_size=0.33, random_state=45)

#     X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
#     X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

#     # number of files, number of time steps , number of features
#     X_train_tf = X_train_norm.reshape(
#         X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2])
#     X_test_tf = X_test_norm.reshape(
#         X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2])

#     model = Sequential()

#     model.add(Conv1D(32, kernel_size=3, activation='relu',
#               input_shape=(X_train_tf.shape[1], X_train_tf.shape[2])))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(2, activation='softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])

#     model.fit(X_train_tf, y_train, batch_size=None, epochs=3,
#               verbose=1, class_weight={0: 1., 1: 10.})

#     loss, acc = model.evaluate(X_test_tf, y_test, batch_size=None, verbose=0)
#     print('Test loss:', loss)
#     print('Test accuracy:', acc)

#     return model, np.mean(X_train), np.std(X_train)


# def get_model2():
#     X_train, X_test, y_train, y_test = train_test_split(
#         spec_matrix_db, y, test_size=0.33, random_state=45)

#     X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
#     X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

#     # reshape to add the channels dimension
#     X_train_norm = np.expand_dims(X_train_norm, axis=3)
#     X_test_norm = np.expand_dims(X_test_norm, axis=3)

#     model = Sequential()

#     model.add(Conv2D(filters=8, kernel_size=(8, 10), strides=(2, 2), activation='relu',
#               input_shape=(X_train_norm.shape[1], X_train_norm.shape[2], X_train_norm.shape[3])))
#     model.add(Flatten())
#     model.add(Dense(3, activation='relu'))
#     model.add(Dense(2, activation='softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])

#     model.fit(X_train_norm, y_train, batch_size=None, epochs=3,
#               verbose=1, class_weight={0: 1., 1: 10.})

#     loss, acc = model.evaluate(X_test_norm, y_test, batch_size=None, verbose=0)
#     print('Test loss:', loss)
#     print('Test accuracy:', acc)

#     return model, np.mean(X_train), np.std(X_train)


# model, mean, std = get_model()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)


# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# with open('mean.pkl', 'wb') as f:
#     pickle.dump(mean, f)

# with open('std.pkl', 'wb') as f:
#     pickle.dump(std, f)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mean.pkl', 'rb') as f:
    mean = pickle.load(f)

with open('std.pkl', 'rb') as f:
    std = pickle.load(f)


wav_root = '../data/audio_1sec/'

labels = pd.read_csv('../labels/audio_1sec.csv',  header=None,
                     names=["path", "yes", "no", "not_sure", "subject_set"])
labels['path'] = labels['path'].astype(str) + '.wav'

labels['res'] = (labels['yes'].astype(int)*1 + labels['not_sure'].astype(int)
                 * 0.5) / (labels['yes'] + labels['no'] + labels['not_sure']) >= 0.5
n = len(labels.index.values.tolist())

y = np.zeros((n, 2))

y[:, 1] = np.array(np.array(labels['res']).astype(int)).astype(int)
y[:, 0] = 1-y[:, 1].astype(int)


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 5
win_len = 0.1
nfft = int(win_len * RATE)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
arduino = serial.Serial('/dev/ttyACM0', 9600)


for i in range(10):
    print("Recording...")
    frames = []
    for j in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Convert the recorded audio to numpy array
    audio_np = np.frombuffer(b"".join(frames), dtype=np.int16)

    # Extract MFCC features
    spec = librosa.feature.mfcc(y=audio_np, sr=RATE, n_mfcc=13,
                                n_fft=nfft*4, hop_length=nfft)

    # Normalize the MFCC features using the mean and standard deviation of training data
    spec_norm = (spec - mean) / std

    # Reshape the MFCC features
    spec_reshaped = spec_norm.reshape(
        1, spec_norm.shape[0], spec_norm.shape[1])

    # Predict the class of the audio sample
    y_pred = model.predict(spec_reshaped)
    y_pred_label = np.argmax(y_pred, axis=1)

    print("Predicted class:", y_pred_label[0])
    arduino.write((str(y_pred_label[0])).encode())
