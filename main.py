from lib import pulse_shaping
from lib import matched_filtering

from lib import mode_preconfiguration

from lib import packet
from lib import CRC

from lib import symbol_mod_QAM as mod
from lib import symbol_demod_QAM as demod

from lib import sync
from sync import coarse_freq_sync, fine_freq_sync, phase_sync

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq, fftshift

#####################################
# Declare Variables
fs = 2.4e9    
Ts = 1.0 / fs          # sampling period in seconds
f0 = 0.0               # homodyne (0 HZ IF)
M = 4                  # oversampling factor
T = M*Ts               # symbol period in seconds
Rs = 1/T               # symbol rate
segment_size = 16      # Might need to alter this to reflect actual length of packet
R = 1                  # Packet Ratio: number of segments contained in our larger OOK packet 
N = R*segment_size     # OOK Packet Length (equals R* segment_size)
L = 8                # pulse shaping filter length. Is this the ideal length
pulse_shape = "rrc"    # type of pulse shaping, also "rect"
scheme = "QAM"         # Modulation scheme 'OOK', 'QPSK', or 'QAM'
preamble_ones_length = 10
alpha = 0.9            # roll-off factor of the RRC matched-filter

#####################################
# Generate preamble and packet
data = np.random.randint(0,2,size=256) 
dataLen = len(data)

key = np.array([1,1,1,1,0,0,1,0,1])

dataPacket = packet.packet_generator(data, key, preamble_ones_length)
packetLen = len(dataPacket)
remainderLen = len(key)-1

################################
# Generate Preamble check for matched filter
known_preamble_bits = packet.preamble_generator(preamble_ones_length)
preamble_length = len(known_preamble_bits)
total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,preamble_length,M,fs)
known_preamble_symbols = mod.symbol_mod(known_preamble_bits, "OOK", len(known_preamble_bits))
known_preamble = np.abs(pulse_shaping.pulse_shaping(known_preamble_symbols, samples_perbit, fs_in, 'rect', 0.9, 8))

payload_start = preamble_length
N = len(known_preamble_bits)

######################
# Generate symbols
baseband_symbols = mod.symbol_mod(dataPacket, scheme, preamble_length)
data_symbols = mod.symbol_mod(data, scheme, 0)

######################
# Pulse Shaping
baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, 0.9, 8)
plt.plot(baseband[200:260], 'r-')
######################
# Simulate Channel

# Add 13 kHz frequency offset
fo = 13e5
Ts = 1/fs
t = np.arange(0, Ts*len(baseband), Ts)
baseband = baseband * np.exp(1j*2*np.pi*fo*t)
# plt.plot(baseband[200:260], 'b-')

# Add phase delay
delay = 0.4
N_taps = 16
n = np.arange(-N//2, N//2)
h = np.sinc(n - delay)
h *= np.hamming(N_taps)
h/= np.sum(h)
baseband = np.convolve(baseband, h)
plt.plot(np.real(baseband[200:260]), 'g-')
plt.plot(np.imag(baseband[200:260]), 'b-')

plt.show()

########################
# Coarse Freq Sync
coarse_frequency = coarse_freq_sync(data, payload_start, samples_perbit, fs_in)

########################
# Fine Freq Sync
carrier_frequency = fine_freq_sync(data, payload_start, preamble_length, samples_perbit, coarse_frequency, fs_in)

########################
# Phase Sync
baseband_signal_I, baseband_signal_Q = phase_sync(data, payload_start, samples_perbit, preamble_length, scheme, carrier_frequency, N, fs_in)

########################
# Matched Filtering

baseband_signal_I_new = np.real(baseband)
baseband_signal_Q_new = np.imag(baseband)

symbols_I = matched_filtering.matched_filtering(baseband_signal_I_new, samples_perbit, fs_in, alpha, L, pulse_shape)
symbols_Q = matched_filtering.matched_filtering(baseband_signal_Q_new, samples_perbit, fs_in, alpha, L, pulse_shape)
plt.plot(symbols_I[200:260], 'b-')
plt.plot(symbols_I[200:260], 'g-')

channel_gain = max(symbols_I)

symbols_I = np.real(symbols_I[0:(len(symbols_I))-L+1] / (channel_gain/3))
symbols_Q = np.real(symbols_Q[0:(len(symbols_Q))-L+1] / (channel_gain/3))

########################
# Demod

buff = np.array([symbols_I,symbols_Q])

demod_bits = demod.symbol_demod(buff, scheme, 1, preamble_length) # gain has to be set to 1
error = CRC.CRCcheck(demod_bits, key)

if error:
    print("Error!")
else:
    print("No error")

########################
# Frame Detect Sync


########################
# BER Calculation

demod_bits = demod_bits[:-remainderLen]
print(np.array_equal(dataPacket[:-remainderLen], demod_bits, equal_nan=False))

