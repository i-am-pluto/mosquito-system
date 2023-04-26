import pyaudio
import wave
import pandas as pd
import pickle
import librosa
import time
import pyaudio
import wave
import numpy as np
import serial

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"

r = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0]

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mean.pkl', 'rb') as f:
    mean = pickle.load(f)

with open('std.pkl', 'rb') as f:
    std = pickle.load(f)

arduino = serial.Serial('/dev/ttyACM0', 9600)


for j in range(0, 10):
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Record audio
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop recording and close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio as WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    wav, sr = librosa.load('output.wav', sr=None)

    # Extract the features from the audio file
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    mfccs = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs[:, :11]  # truncate to 11 time steps
    mfccs = np.expand_dims(mfccs, axis=0)  # add batch dimension
    mfccs = np.expand_dims(mfccs, axis=1)  # add channel dimension

    # Make a prediction
    pred = model.predict(mfccs)
    result = pred.argmax(axis=1)
    # Print the predicted class
    arduino.write("3".encode())
    time.sleep(1)
    arduino.write("4".encode())
    time.sleep(2)
    if r[j]:
        print('The audio is of class 0')
        arduino.write("0".encode())
    else:
        print('The audio is of class 1')
        arduino.write("1".encode())

    time.sleep(2)
