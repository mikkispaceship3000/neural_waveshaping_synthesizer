import wave 
import librosa 
import numpy as np 
import torch 
from scipy.io import wavfile 

n_fft = 1024
HOP_LENGTH = n_fft // 4

def get_features(input_waveform): 
	#output = list of f0 for each time step
	#output = list of spectral features for each time step 
	
	signal, sampling_rate = librosa.load(input_waveform, sr=None)
	f0s, smth, probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sampling_rate, hop_length = HOP_LENGTH)
	# Replace NaN values with 0 for unvoiced frames
	f0s = np.nan_to_num(f0s)
	# print(f"first 10 f0s: {f0s[:10]}")
	features = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, hop_length=HOP_LENGTH, n_fft=n_fft)
	stftfeatures = librosa.stft(signal, hop_length = HOP_LENGTH, n_fft=n_fft)
	
	# print(f"first 10 features: {features[:10]}")
	# print(f"sampling rate: {sampling_rate}")
	# print(f"shape of f0s: {f0s.shape}")
	# print(f"shape of features: {features.shape}")
	duration = librosa.get_duration(y=signal, sr=sampling_rate, n_fft=n_fft, hop_length = HOP_LENGTH)
	# print(f"duration: {duration}")
	num_samples = len(signal)
	if sampling_rate != 44100:
		print("Not sure what to do with this file yet, sample rate not as expected")
		exit(1)

	tonal_features = librosa.feature.tonnetz(y = signal, sr = sampling_rate, hop_length=HOP_LENGTH)
	return f0s, features, sampling_rate, duration, tonal_features, stftfeatures


	