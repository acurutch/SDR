# FOR TESTING STRAIGHT UP MOD/DEMOD
#############
# FOR INCORPORATING PULSE SHAPING
import pulse_shaping
import matched_filtering

import mode_preconfiguration

import packet
import CRC

import symbol_mod_QAM as mod
import symbol_demod_QAM as demod

import packet
import CRC

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq, fftshift

import time

#####################################
# Declare Variables
fs = 2.4e9    
Ts = 1.0 / fs          # sampling period in seconds
f0 = 0.0               # homodyne (0 HZ IF)
M = 4 # oversampling factor
T = M*Ts               # symbol period in seconds
Rs = 1/T               # symbol rate
segment_size = 16    # Might need to alter this to reflect actual length of packet
R = 1               # Packet Ratio: number of segments contained in our larger OOK packet 
N = R*segment_size     # OOK Packet Length (equals R* segment_size)
L = 8                  # pulse shaping filter length
# Is this the ideal length
pulse_shape = "rrc" # type of pulse shaping, also "rect"
scheme = "QAM"    # Modulation scheme 'OOK', 'QPSK', or 'QAM'
preamble_ones_length = 10
alpha = 0.9 #roll-off factor of the RRC matched-filter

#####################################
# Generate preamble and packet
# data = np.random.randint(0,2,size=256) 
# data = np.append(np.zeros(3), np.ones(1))
# data = np.append(data, np.zeros(1))
# data = np.append(data, np.ones(2))
# data = np.append(data, np.zeros(1))
# data = np.append(data, np.ones(1))
# data = np.append(data, np.zeros(1))
# data = np.append(data, np.ones(2))
# data = np.append(data, np.zeros(4))

data = np.random.randint(0,2,size=256) 

print("data: ")
print(data)
dataLen = len(data)

key = np.array([1,1,1,1,0,0,1,0,1])

dataPacket = packet.packet_generator(data, key, preamble_ones_length)
print("data packet: ")
print(dataPacket)
packetLen = len(dataPacket)
remainderLen = len(key)-1

################################
# Generate Preamble check for matched filter
known_preamble_bits = packet.preamble_generator(preamble_ones_length)
preamble_length = len(known_preamble_bits)
total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,preamble_length,M,fs)
known_preamble_symbols = mod.symbol_mod(known_preamble_bits, "OOK", len(known_preamble_bits))
print("known preamble symbols: ")
print(known_preamble_symbols)
known_preamble = np.abs(pulse_shaping.pulse_shaping(known_preamble_symbols, samples_perbit, fs_in, 'rect', 0.9, 8))

######################
# Generate symbols
baseband_symbols = mod.symbol_mod(dataPacket, scheme, preamble_length)
print("baseband symbols: ")
print(baseband_symbols)
data_symbols = mod.symbol_mod(data, scheme, 0)
print("data symbols: ")
print(data_symbols)

######################
# Pulse Shaping
baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, 0.9, 8)

########################
# Matched Filtering

baseband_signal_I_new = np.real(baseband)
baseband_signal_Q_new = np.imag(baseband)

symbols_I = matched_filtering.matched_filtering(baseband_signal_I_new, samples_perbit, fs_in, alpha, L, pulse_shape)
symbols_Q = matched_filtering.matched_filtering(baseband_signal_Q_new, samples_perbit, fs_in, alpha, L, pulse_shape)

channel_gain = max(symbols_I)

symbols_I_1 = np.real(symbols_I[0:len(baseband_symbols)] / (channel_gain/max(baseband_symbols)))
symbols_Q_1 = np.real(symbols_Q[0:len(baseband_symbols)] / (channel_gain/max(baseband_symbols)))

symbols_I_2 = np.real(symbols_I[L-1:(len(symbols_I))-L] / (channel_gain/max(baseband_symbols)))
symbols_Q_2 = np.real(symbols_Q[L-1:(len(symbols_Q))-L] / (channel_gain/max(baseband_symbols)))

symbols_I = np.real(symbols_I[0:(len(symbols_I))-L+1] / (channel_gain/max(baseband_symbols)))
symbols_Q = np.real(symbols_Q[0:(len(symbols_Q))-L+1] / (channel_gain/max(baseband_symbols)))

print("Symbols I:")
print(symbols_I)

print("Symbols Q:")
print(symbols_Q)


##################
# Plot I symbols

fig, axs = plt.subplots(2, 2)

fig.suptitle("I Symbols")

axs[0, 0].plot(np.real(baseband_symbols), 'r-')
axs[0, 0].set_title("Before Pulse Shaping")
axs[0, 1].plot(baseband_signal_I_new, 'b-')
axs[0, 1].plot(np.real(baseband_symbols), 'r-')
axs[0, 1].set_title("After Pulse Shaping")
axs[1, 0].plot(symbols_I)
axs[1, 0].plot(np.real(baseband_symbols), 'r-')
axs[1, 0].set_title("After Matched Filtering")

axs[1,1].plot(symbols_I[0:len(known_preamble_symbols)],symbols_Q[0:len(known_preamble_symbols)], 'ro')
axs[1,1].plot(symbols_I[len(known_preamble_symbols):],symbols_Q[len(known_preamble_symbols):], 'bo')
axs[1,1].legend(['Preamble Symbols', 'Data Symbols'])
axs[1,1].set_title("IQ Plot")

major_ticks = np.arange(-4, 5, 2)
axs[1,1].set_xticks(major_ticks)
axs[1,1].set_yticks(major_ticks)

axs[1,1].minorticks_on()

minor_ticks = np.arange(-4, 5, 1)
axs[1,1].set_xticks(minor_ticks, minor=True)
axs[1,1].set_yticks(minor_ticks, minor=True)

axs[1,1].grid(which='minor', linestyle=':', alpha=0.2)
axs[1,1].grid(which='major', linestyle='-', alpha=0.4)

axs[1,1].set_xlim([-4, 4])
axs[1,1].set_ylim([-4, 4])

fig.show()

##################
# Plot Q symbols

fig2, axs2 = plt.subplots(2, 2)

fig2.suptitle("Q Symbols")

axs2[0, 0].plot(np.imag(baseband_symbols), 'r-')
axs2[0, 0].set_title("Before Pulse Shaping")
axs2[0, 1].plot(baseband_signal_I_new, 'b-')
axs2[0, 1].plot(np.imag(baseband_symbols), 'r-')
axs2[0, 1].set_title("After Pulse Shaping")
axs2[1, 0].plot(symbols_Q)
axs2[1, 0].plot(np.imag(baseband_symbols), 'r-')
axs2[1, 0].set_title("After Matched Filtering")

axs2[1,1].plot(symbols_I[0:len(known_preamble_symbols)],symbols_Q[0:len(known_preamble_symbols)], 'ro')
axs2[1,1].plot(symbols_I[len(known_preamble_symbols):],symbols_Q[len(known_preamble_symbols):], 'bo')
axs2[1,1].legend(['Preamble Symbols', 'Data Symbols'])
axs2[1,1].set_title("IQ Plot")

major_ticks = np.arange(-4, 5, 2)
axs2[1,1].set_xticks(major_ticks)
axs2[1,1].set_yticks(major_ticks)

axs2[1,1].minorticks_on()

minor_ticks = np.arange(-4, 5, 1)
axs2[1,1].set_xticks(minor_ticks, minor=True)
axs2[1,1].set_yticks(minor_ticks, minor=True)

axs2[1,1].grid(which='minor', linestyle=':', alpha=0.2)
axs2[1,1].grid(which='major', linestyle='-', alpha=0.4)

axs2[1,1].set_xlim([-4, 4])
axs2[1,1].set_ylim([-4, 4])

fig2.show()

########################
# FFT

fig3, ax = plt.subplots(1,1)

before_ps = baseband_symbols
after_ps = baseband

y = fft(before_ps)
xf = fftfreq(len(y), 0.1)
xf = fftshift(xf)
yplot = fftshift(y)

ax.plot(xf, 1.0/100 * np.abs(yplot), 'r-')

y = fft(after_ps)
xf = fftfreq(len(y), 0.1)
xf = fftshift(xf)
yplot = fftshift(y)

ax.plot(xf, 1.0/100 * np.abs(yplot), 'b-')

ax.grid()
ax.legend(['Before Pulse Shaping', 'After Pulse Shaping'])
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
ax.set_title('Pulse Shaping')
fig3.show()

########################
# Demod

# For test
# baseband_signal_I_new = np.real(baseband_symbols)
# baseband_signal_Q_new = np.imag(baseband_symbols)

buff = np.array([symbols_I,symbols_Q])

demod_bits = demod.symbol_demod(buff, scheme, 1, preamble_length) # gain has to be set to 1
error = CRC.CRCcheck(demod_bits, key)

if error:
    print("Error!")
else:
    print("No error")

demod_bits = demod_bits[:-remainderLen]
print("Demod bits: ")
print(demod_bits)
print("Data Packet: ")
print(dataPacket)
print(np.array_equal(dataPacket[:-remainderLen], demod_bits, equal_nan=False))

