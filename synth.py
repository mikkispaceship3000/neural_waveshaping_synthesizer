import getfeatures 
import sourcefilter 
from scipy.io import wavfile #is this not being used
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch


def pitch_extract(input_waveform):
	"""
    Extracts the pitch (F0) from the input waveform and generates a sine-based excitation signal.
    
    Returns:
    - pitch_extract_wave: Sine-based waveform from fundamental frequencies (f0s)
    """
	f0s, features, sampling_rate, duration, tonal_features, stftfeatures = getfeatures.get_features(input_waveform)	 
	pitch_extract_wave = sourcefilter.source_module(f0s)
	return pitch_extract_wave

def resynthesize(input_waveform): 
	"""
    Resynthesizes the original waveform by:
    - generating excitation from pitch extracting function
    - applying the learned spectral filter based on original spectral features
    
    Returns:
    - output_wave: resyntheized version of OG waveform
    """
	f0s, features, sampling_rate, duration, tonal_features, stftfeatures = getfeatures.get_features(input_waveform)	 
	excitation_signal = pitch_extract(input_waveform)
	output_wave = sourcefilter.filter_module(excitation_signal, stftfeatures)
	return output_wave

def pitch_shift(input_waveform, semitones):
	 """
    Shifts the pitch of the input waveform by a specified number of semitones:
    - f0s are scaled accordingly
    - excitation is regenerated with shifted pitch
    - original spectral envelope is used
    
    Returns:
    - pitchshift_wave: pitch-shifted version of the input
    """
	f0s, features, sampling_rate, duration, tonal_features, stftfeatures = getfeatures.get_features(input_waveform)	 
	shifted_f0s = f0s * (2 ** (semitones / 12))
	pitch_shift_excitation = sourcefilter.source_module(shifted_f0s)
	pitchshift_wave = sourcefilter.filter_module(pitch_shift_excitation, stftfeatures)
	return pitchshift_wave

def timbre_transfer(input_waveform_env, input_waveform_pitch):
	 """
    Performs timbre transfer:
    - uses pitch from one waveform (input_waveform_pitch)
    - applies spectral envelope (timbre) from another (input_waveform_env)
    
    Returns:
    - output_wav: waveform with source pitch and target timbre
    """
	excitation = pitch_extract(input_waveform_pitch)
	f0s, features, sampling_rate, duration, tonal_features, stftfeatures = getfeatures.get_features(input_waveform_env)	 
	output_wav = sourcefilter.filter_module(excitation, stftfeatures)
	return output_wav

def play(wave, name): 
	   """
    Saves the generated waveform to comp as a .wav file for playback.
    
    Parameters:
    - wave: audio waveform (Tensor or ndarray)
    - name: filename for saving
    """
	if isinstance(wave, torch.Tensor):
		toplay = wave.squeeze().detach().numpy()
	else: 
		toplay = wave.squeeze()

	wavfile.write(f'{name}.wav', 44100, (toplay * 32767).astype(np.int16))

#optional
def show(features): 
	plt.figure()
	features = librosa.power_to_db(features, ref=np.max)
	librosa.display.specshow(features, sr=44100, hop_length = n_fft // 4, x_axis='time', y_axis='mel', cmap='magma')
	plt.colorbar(format='%+2.0f dB')  
	plt.title('Mel-Spectrogram')
	plt.tight_layout()
	plt.show()
