import random
from scipy.io import wavfile
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from p5 import *
from socketio import Client

import keras
from keras import layers, initializers
import numpy as np
import tensorflow
import tensorflow as tf
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile
import matplotlib.pyplot as plt

sio = Client()
sio.connect('http://localhost:8080')
print('SID:', sio.sid)

files = glob.glob('./datasets/AKWF/*.wav')
print('There are:',len(files), 'files.')
data = np.empty((1,600)) #create empty array with same shape as data will be
print(data.shape)
mms = MinMaxScaler(feature_range = (-1 ,1))
for file in files:
	samplerate, wav_data = wavfile.read(file)
	
	if wav_data.shape == (600,):

		wav_data = wav_data.reshape(wav_data.shape[0],1)
		t = mms.fit_transform(wav_data)
		t = t.reshape(1,t.shape[0])
		t = np.nan_to_num(t)
	else:
		print('Encountered multi-channel wav:', file)
	data = np.append(data,t,axis = 0) #append along correct axis

# data = data[-100::] #only 100 of the 4000

encoder = keras.models.load_model("WORKING_encoder.h5")
decoder = keras.models.load_model("WORKING_decoder.h5")


encoded_data = encoder.predict(data)

def setup():
	size(600,600)
	background(255)


	
	pass

def draw():
	background(255)
	stroke(255,0,0,255)
	stroke_weight(5)
	# print(mouse_x, mouse_y)
	point(mouse_x, mouse_y)
	mmxs = MinMaxScaler(feature_range = (0,1))
	mxs = mouse_x / 600 #normalized to width and height
	mys = mouse_y / 600 
	# print(mxs, mys)
	latent_waveform = decoder.predict(np.array([mxs, mys]).reshape(1,2))[0]
	# print(latent_waveform)
	scaler = MinMaxScaler(feature_range = (-1,1))
	latent_waveform = scaler.fit_transform(latent_waveform)
	# for i,x in enumerate(encoded_data):
		
	# 	stroke(0,0,0,255)
	# 	point(x[0]*600,x[1]*600)

	stroke_weight(2)
	stroke(0,0,0,255)
	for i,p in enumerate(latent_waveform):

		x = i
		y = 300 + p * 300
		# point(x,y)
		line(x,y, i-1,300 +latent_waveform[i-1] * 300)	


	it = iter(latent_waveform.tolist())
	res_dict = dict(zip(range(len(latent_waveform)),it))
	sio.emit('dictionary', res_dict)

if __name__ == '__main__':
	run()