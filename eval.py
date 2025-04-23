import librosa
import numpy as np 
import matplotlib.pyplot as plt 

def compute_spectrogram(audio): 
	#compute spectrogram from audio waveform
	#returns scaled spectrogram for db 
	stft = librosa.stft(audio, n_fft=1024, hop_length=256)
	spectrogram = np.abs(stft) #magnitude
	spectrogram_db = librosa.amplitude_to_db(spectrogram, ref = np.max)
	return spectrogram_db

def plot_spectrogram(spec, title): 
	#display spectro
	plt.figure(figsize=(10,4))
	librosa.display.specshow(spec, sr=44100, hop_length=256, x_axis = 'time', y_axis = 'log', cmap = 'magma')
	plt.colorbar(format='%+2.0f dB')
	plt.title(title)
	plt.tight_layout()
	plt.show()

def compare_spectrograms(audio1str, audio2):
	#plot spectrograms side by side
	audio1, _ = librosa.load(audio1str, sr=None, dtype=np.float32)
	spec1 = compute_spectrogram(audio1)
	spec2 = compute_spectrogram(audio2)
	plt.subplot(2, 1, 1)
	librosa.display.specshow(spec1, sr=44100, hop_length=256, x_axis = 'time', y_axis = 'log', cmap = 'magma')
	plt.colorbar(format='%+2.0f dB')
	plt.title("Input Audio Spectrogram")

	plt.subplot(2, 1, 2)
	librosa.display.specshow(spec2, sr=44100, hop_length=256, x_axis = 'time', y_axis = 'log', cmap = 'magma')
	plt.colorbar(format='%+2.0f dB')
	plt.title("Output Audio Spectrogram")

	plt.tight_layout()
	plt.show()

def plot_spectral_difference(inputaudiostr, audio2):
	#find difference between two spectros and plot
	audio1, _ = librosa.load(inputaudiostr, sr=None, dtype=np.float32)
	spec1 = compute_spectrogram(audio1)
	spec2 = compute_spectrogram(audio2)
	
	#make sure the spectrograms are the same size
	min_time = min(spec1.shape[1], spec2.shape[1])
	spec1 = spec1[:, :min_time]
	spec2 = spec2[:, :min_time]
	diff_spec = spec1-spec2

	plt.figure(figsize=(10,4))
	librosa.display.specshow(diff_spec, sr=44100, hop_length=256, x_axis = 'time', y_axis = 'log', cmap = 'magma')
	plt.colorbar(format='%+2.0f dB')
	plt.title("Spectral Difference")
	plt.tight_layout()
	plt.show()	

def log_spectral_difference(audio1, audio2):
	#compute lsd scores
	if type(audio1) == str:
		audio1, _ = librosa.load(audio1, sr=None, dtype=np.float32)
	
	spec1 = compute_spectrogram(audio1)
	spec2 = compute_spectrogram(audio2)

	#make sure the spectrograms are the same size
	min_time = min(spec1.shape[1], spec2.shape[1])
	spec1 = spec1[:, :min_time]
	spec2 = spec2[:, :min_time]

	#avoid NaNs, get magnitude spectra  
	spec1 = np.abs(spec1) + 1e-8
	spec2 = np.abs(spec2) + 1e-8

	log_spec_in = np.log1p(spec1)
	log_spec_out = np.log1p(spec2)

	lsd = np.mean(np.sqrt(np.sum((log_spec_in - log_spec_out) ** 2, axis=0)))
	print(f"Log Spectral Distance: {lsd:.4f}")
	return lsd

def signal_to_noise(audio): 
	#find snr scores of signal
	if type(audio) == str:
		audio, _ = librosa.load(audio, sr=None, dtype=np.float32)
	
	#noise is difference between smoothed version of the audio and the audio og version 
	smooth = np.convolve(audio, np.ones(5)/5, mode = 'same')
	noise = audio - smooth 

	signal_power = np.mean(audio ** 2)
	noise_power = np.mean(noise ** 2)

	if noise_power == 0: 
		return float('inf')

	#snr in decibels formula
	snr = 10 * np.log10(signal_power / noise_power)
	return snr

