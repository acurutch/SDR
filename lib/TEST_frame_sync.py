import numpy as np
from scipy import signal
import packet
import mode_preconfiguration
import pulse_shaping
import symbol_mod_QAM as mod
import pyfftw
import sync

import matplotlib.pyplot as plt

fs = 75000  
pulse_shape = "rrc"    # type of pulse shaping, also "rect"
scheme = "QAM"         # Modulation scheme 'OOK', 'QPSK', or 'QAM'
preamble_ones_length = 100
M = 4
payloadSize = 256

#####################################
# Generate preamble and packet
data = np.random.randint(0,2,size=payloadSize) 
dataLen = len(data)

key = np.array([1,1,1,1,0,0,1,0,1])

dataPacket = packet.packet_generator(data, key, preamble_ones_length)
packetLen = len(dataPacket)
remainderLen = len(key)-1

#####################################
# Generate known preamble information

known_preamble_bits = packet.preamble_generator(preamble_ones_length)
N = len(known_preamble_bits)

preamble_length = len(known_preamble_bits)
total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,preamble_length,M,fs)
known_preamble_symbols = mod.symbol_mod(known_preamble_bits, "OOK", len(known_preamble_bits))
known_preamble = np.abs(pulse_shaping.pulse_shaping(known_preamble_symbols, samples_perbit, fs_in, 'rect', 0.9, 8))

######################
# Generate symbols
baseband_symbols = mod.symbol_mod(dataPacket, scheme, preamble_length)
data_symbols = mod.symbol_mod(dataPacket[preamble_length:], scheme, 0)
known_data = pulse_shaping.pulse_shaping(data_symbols, samples_perbit, fs_in, 'rrc', 0.9, 8)

######################
# Pulse Shaping
baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, 0.9, 8)

######################
# Simulate Channel

# Add 13 kHz frequency offset
fo = 130e5
Ts = 1/fs
t = np.arange(0, Ts*len(baseband), Ts)
test_data = baseband * np.exp(1j*2*np.pi*fo*t)

# Add phase delay
delay = 0.8
N_taps = N
n = np.arange(-N//2, N//2)
h = np.sinc(n - delay)
h *= np.hamming(N_taps)
h/= np.sum(h)
test_data = np.convolve(test_data, h)

test_data = np.abs(np.real(test_data)) + np.abs(np.imag(test_data))

matched_filter_coef = np.flip(known_preamble)
# crosscorr = signal.fftconvolve(abs(test_data),matched_filter_coef)
# crosscorr = signal.fftconvolve(np.power(3,test_data),matched_filter_coef)
crosscorr = signal.fftconvolve(test_data,matched_filter_coef)

# peak_indices, _ = signal.find_peaks(crosscorr, height =  peak_threshold, distance = int(0.8*total_samples))
# this finds ALL packets contained in a given buffer

peak_threshold = 400
peak_indices, _ = signal.find_peaks(np.real(crosscorr), height = peak_threshold, distance = 1)

max_peak = peak_indices[0]

for peak in peak_indices:
    if ((crosscorr[peak] > crosscorr[max_peak]) and (np.abs(np.imag(crosscorr[peak])) < 0.1)):
        max_peak = peak
        
if np.abs(np.imag(np.max(crosscorr))) < 0.1:
    idx = np.array(crosscorr).argmax()
else:
    idx = max_peak



## Whole thing doesn't work, frequency offset throws it off

# idx = np.array(crosscorr).argmax() # can't use if there are multiple packets in a dataframe

recoveredData = test_data[idx:]

plt.plot(matched_filter_coef, 'k')
plt.plot(test_data, 'g')
plt.plot(crosscorr/80, 'b')
plt.stem(idx, 10, 'r')
plt.show()

plt.plot(known_data, 'k')
plt.plot(recoveredData, 'g')
plt.plot(test_data, 'b')
plt.stem(idx, 4, 'r') # this position is off
# also, some issue here with filter delay. the delay of the pulse-shaped entire packet is longer than that of the pulse-shaped data
# this means that there may be some issue with matched filtering just the data, or the entire payload starting at the first preamble signal
# (excluding small delay at the front (if that exists?)))
plt.show()


##################################
# Testing frequency correction

#Estimate the frequency offset

num_of_bits_fft = 180
segments_of_data_for_fft = data
#Determining number of FFT points (this decides the resolution of frequency bins for CFO estimation)
original_fft_point = np.power(2,20)
coarse_fft_point = original_fft_point/8  #2^X point FFT

#Peform FFT on a segment of samples recevied
#Hint 1: You will want to use library functions including: abs() and pyfftw.interfaces.scipy_fftpack.fft()
# (which is a faster version of FFT function compared with SciPy/NumPy implementation)
#Hint 2: Our goal is to peform FFt on array of samples named "segments_of_data_for_fft" with
# "coarse_fft_point" number of points

#spectrum = "Perform FFT on the array of samples here"
spectrum = abs(pyfftw.interfaces.scipy_fftpack.fft(segments_of_data_for_fft, coarse_fft_point)) 

#FFT shift so that DC component is in the middle of the array of FFT result                
spectrum = np.fft.fftshift(spectrum)

#Determine the index of the peak magnitude in FFT result
#Hint: You may find np.argmax() useful
#peak_position = "Determine the location/index of the peak FFT magnitude"
peak_position = np.argmax(spectrum)

#Obtain the estimated carrier frequency offset
carrier_frequency = (peak_position-len(spectrum)/2) / len(spectrum) * fs_in

baseband_signal_I, baseband_signal_Q = sync.phase_sync(data, 0, samples_perbit, preamble_length, scheme, carrier_frequency, N, fs_in)
