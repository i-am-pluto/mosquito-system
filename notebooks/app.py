import pandas as pd
import pickle
import librosa
import random
import time
import numpy as np
import serial


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


# Load the audio file you want to predict
wav_root = '../data/audio_1sec/'
wav_path = wav_root + labels["path"]
win_len = 0.1


# arduino = serial.Serial('/dev/ttyACM0', 9600)

# replace 'COM4' with the serial port where your Arduino is connected
arduino = serial.Serial('/dev/ttyACM0', 9600)

for i in range(10):
    index = random.randint(0, len(wav_path)-1)
    wav_file = wav_path[index]
    wav, sr = librosa.load(wav_file, sr=None)

    # Extract the features from the audio file
    nfft = int(win_len * sr)
    spec = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=13, n_fft=nfft*4, hop_length=nfft)
    spec_norm = (spec - mean) / std

    # Reshape the data to match the input shape of the model
    spec_norm_reshaped = spec_norm.reshape(
        1, 1, spec_norm.shape[0], spec_norm.shape[1])

    # Make a prediction
    pred = model.predict(spec_norm_reshaped)

    r = (labels.iloc[index]['yes'].astype(
        int)*1 + labels.iloc[index]['not_sure'].astype(int)*0.5) / (labels.iloc[index]['yes'] + labels.iloc[index]['no'] + labels.iloc[index]['not_sure']) >= 0.5
    result = pred.argmax(axis=1)

    arduino.write("3".encode())
    time.sleep(1)
    arduino.write("4".encode())
    time.sleep(2)
    if r == False:
        print('The audio file', wav_file, 'is of class 0', ', Label ', (labels.iloc[index]['yes'].astype(
            int)*1 + labels.iloc[index]['not_sure'].astype(int)*0.5) / (labels.iloc[index]['yes'] + labels.iloc[index]['no'] + labels.iloc[index]['not_sure']) >= 0.5)
        arduino.write("0".encode())
    else:
        print('The audio file', wav_file, 'is of class 1', ', Label ', (labels.iloc[index]['yes'].astype(
            int)*1 + labels.iloc[index]['not_sure'].astype(int)*0.5) / (labels.iloc[index]['yes'] + labels.iloc[index]['no'] + labels.iloc[index]['not_sure']) >= 0.5)
        arduino.write("1".encode())

    time.sleep(2)
    # Display the playable audio file

while (True):
    c = input()
    arduino.write(c.encode())
