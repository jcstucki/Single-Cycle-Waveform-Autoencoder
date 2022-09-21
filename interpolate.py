import random
from scipy.io import wavfile
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from p5 import *
from socketio import Client

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
	else:
		print('Encountered multi-channel wav:', file)
	data = np.append(data,t,axis = 0) #append along correct axis


latent_array = np.zeros_like(data[0,:])

def setup():
	size(600,600)
	background(255)

	pass

def draw():
	background(255)
	
	lerp_amount = 0.5
	waveform = random.choice(data)
	for i,destination_point in enumerate(waveform):
		source_point = latent_array[i]
		latent_array[i] = lerp(source_point,destination_point,lerp_amount)

	for i,p in enumerate(latent_array):
		x = i
		y = 300 + p * 300
		point(x,y)
		line(x,y, i-1,300 +latent_array[i-1] * 300)

	it = iter(latent_array)
	res_dict = dict(zip(range(len(latent_array)),it))

	sio.emit('dictionary', res_dict)

if __name__ == '__main__':
	run()