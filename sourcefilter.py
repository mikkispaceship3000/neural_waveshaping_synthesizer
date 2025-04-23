import wave 
import librosa 
import numpy as np
import torch
import scipy
from scipy.io import wavfile
import synth

n_fft = 1024
hop_length = n_fft //4

def source_module(input_f0s): 
	#input_f0s = list of fundamental frequencies at each time step
	#output = excitation signal (waveform made up of sine waves with learned f0s)

	#to include in the generated excitation signal
	NUM_HARMONICS = 25

	#upsampling the input for as many frames as there are in each sample
	#adding a dimension in front of the input list, and multiplying the length by 512, the hop length to get the accurate number of samples in the waveform (vs. the number of frames)
	input_f0s_tensor = torch.tensor(input_f0s, dtype=torch.float32).unsqueeze(0)
	f0s = torch.repeat_interleave(input_f0s_tensor, hop_length, dim=1)
	#shape of f0s: [1, numberofsamples]
	amplitude = torch.tensor([1.0 / (i + 1) for i in range(NUM_HARMONICS)], dtype=torch.float32)

	#create a phase to multiply the pitch
	time_steps = torch.arange(f0s.size(1), dtype=torch.float32) / 44100
	fundamental_phase = 2 * np.pi * torch.cumsum(f0s.squeeze(0) * time_steps, dim=0)
	phi = torch.zeros(NUM_HARMONICS, f0s.size(1))
	for i in range(NUM_HARMONICS): 
		phi[i] = (i+1) * fundamental_phase

	#output shape = [numberofsamples]:
	output = torch.zeros_like(f0s)

	#create harmonics based on f0s, 
	#multiply amplitude by signal and add to output 
	for i in range(NUM_HARMONICS):
		#scale f0s by harmonic index 
		harmonic_f0s = f0s * (i+1)
		output += (amplitude[i] * (signal(harmonic_f0s, phi[i])))
	
	#tanh just to make sure output is audible
	output = torch.tanh(output)
	return output

def signal(frequency, phi):
	#choosing semi-arbitrary scaling factors 
	sigma = 0.0003
	alpha = 0.1
	
	#generate noise 
	noise = torch.normal(0., sigma, size=frequency.shape)

	#generate sine wave with added noise
	outplus_noise = alpha * torch.sin(torch.cumsum(2. * np.pi * frequency / 44100, dim=1)+phi) + noise

	#apply conditions that is silent when frequency is 0: 
	excitation = torch.where(frequency > 0, outplus_noise, torch.zeros_like(frequency))
	return excitation

def filter_module(excitation, features):
	#generate a new signal given the original features and the excitation signal as a starting point
	#new signal recreates original signal 
	#input = excitation:pitch-based sine waveform from source module, features:spectral features from STFT
	#output = waveform shaped by learned spectral envelope 

	OGmagnitude = np.abs(features)
	#normalization 
	envelope = OGmagnitude / (np.max(OGmagnitude) + 1e-8) #the 1e-8 is to avoid division by 0 

	#convert envelope to tensor for processing
	envelope_tensor = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0)
	#simple nn to predict gains
	spectral_shaper = train_simple_nn(envelope_tensor)
	#make envelope into local windows
	with torch.no_grad(): 
		padded = torch.nn.functional.pad(envelope_tensor, (0, 0, 1, 1))
		windows = padded.unfold(1, 3, 1)
		gains = spectral_shaper(windows.permute(0,2,1,3)).transpose(1, 2).squeeze(0).squeeze(-1).numpy()

	#multiply original envelope by learned gains
	transformed_envelope = envelope * gains

	#apply transformed envelope to the excitation signal 
	return combine_feats_and_excitation(transformed_envelope, excitation)


def combine_feats_and_excitation(envelope, excitation): 
	#multiply envelope by excitation signal
	#input = envelope after processing, excitation from source module
	#output = resynthesized waveform

	excitation = excitation.squeeze().detach().cpu().numpy()
	#getting features from the excitation
	excitation_stft = librosa.stft(excitation, n_fft=n_fft, hop_length=hop_length, win_length = n_fft, window = 'hann')
	
	#if for some reason the shapes do not align, resize via interpolation
	if envelope.shape != excitation_stft.shape: 
		#linear interpolation
		zoom_factors = (excitation_stft.shape[0] / envelope.shape[0], excitation_stft.shape[1] / envelope.shape[1])
		envelope = scipy.ndimage.zoom(envelope, zoom_factors, order=1)

	#multiply envelope with excitation features to shape 
	combined_stft = envelope * excitation_stft
	#inverse stft to convert back into time-domain
	output = librosa.istft(combined_stft, hop_length=hop_length, n_fft=n_fft, win_length=n_fft, window='hann') 
	return output * 8.0 #boosting gain


class simpleNN(torch.nn.Module):
	#spectral shaper 
	#predicts frequency gain adjustments based on local window info

	def __init__(self):
		super().__init__()
		self.inputlayer = torch.nn.Linear(3, 16) #3 to 5
		#input 3, output 10.
		#input [current bin, neighbor low, neighbor high]
		self.outputlayer = torch.nn.Linear(16, 1)
		#output = 1, gain adjustment
	def forward(self, x):
		x = torch.relu(self.inputlayer(x))
		x = self.outputlayer(x)
		return x 


def train_simple_nn(envelope, n_epochs=200):
	#trains nn to learn eq-like frequency shaping function on top of patterns from input envelope

	#create context windows (current bin + neighbors)
	padded_env = torch.nn.functional.pad(envelope, (0, 0, 1, 1))#pad frequency dimensions
	windows = padded_env.unfold(1, 3, 1) #batch, freq, time,3 
	batch, freq_bins, time_frames, _ = windows.shape
	
	model = simpleNN()

	#create targets 
	freqs = librosa.fft_frequencies(sr = 44100, n_fft=n_fft)
	targets = torch.ones(freq_bins) #1d 
	for i, j in enumerate(freqs): 
		#reduce muddinesss
		if 200 <= j <= 500:
			targets[i] *= 0.7 
		#more clarity pls 
		elif 2000 <= j <= 5000: 
			targets[i] *= 1.4 
		#add air if freq is way high (boosting very high freqs for brightness)
		elif j > 8000: 
			targets[i] *= 1.2

	#expand shape to match input
	targets = targets.unsqueeze(0).unsqueeze(-1).expand(-1,-1,time_frames)
	#set up optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
	#gonna use small training loop 
	for i in range(n_epochs):
		optimizer.zero_grad()

		inputs = windows.permute(0, 2, 1, 3)
		flat_in = inputs.reshape(-1, 3)

		flat_gains = model(flat_in)
		gains = flat_gains.view(batch, time_frames, freq_bins)
		
		#MSE loss between predicted gains and target gains
		loss = torch.nn.functional.mse_loss(gains, targets.transpose(1, 2))
		loss.backward()
		optimizer.step()
	
	return model 


