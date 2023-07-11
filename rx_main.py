from sync import coarse_freq_sync, fine_freq_sync, phase_sync
import numpy as np
import matched_filtering
import symbol_demod_QAM as demod
import CRC
import packet

# Declare Asynchronous Packet Detection (from socket)

# Main receive routine, contains all signal processing blocks
# Delcare Variables
M = 4 # oversampling factor
samples_perbit = int(M)
fs_in = 2.4e9
L = 8                  # pulse shaping filter length
# Is this the ideal length
pulse_shape = "rrc" # type of pulse shaping, also "rect"
scheme = "QAM"    # Modulation scheme 'OOK', 'QPSK', or 'QAM'
preamble_ones_length = 10
alpha = 0.9 #roll-off factor of the RRC matched-filter

preamble_ones_length = 10
known_preamble_bits = packet.preamble_generator(preamble_ones_length)
preamble_length = len(known_preamble_bits)
payload_start = preamble_length

N = preamble_length # is this right?

# Get results from Asynchronous Packet Detection (from socket)
data = getData()

# For simulation - generate data with frequency and phase offset. Still need to write frame detection and
# asynchronous packet detection. Also need to look into socket programming. Finally, figure out if you are going to use
# USRP or Sidekiq, and look at API functions. May need to write python wrapper for Sidekiq functions

# Coarse Frequency Sync
coarse_frequency = coarse_freq_sync(data, payload_start, samples_perbit, fs_in)

# Fine Frequency Sync
carrier_frequency = fine_freq_sync(data, payload_start, preamble_length, samples_perbit, coarse_frequency, fs_in)

# Phase Sync
baseband_signal_I, baseband_signal_Q = phase_sync(data, payload_start, samples_perbit, preamble_length, scheme, carrier_frequency, N, fs_in)

# Matched Filtering
symbols_I = matched_filtering.matched_filtering(baseband_signal_I, samples_perbit, fs_in, alpha, L, pulse_shape)
symbols_Q = matched_filtering.matched_filtering(baseband_signal_Q, samples_perbit, fs_in, alpha, L, pulse_shape)

channel_gain = max(symbols_I)

# Correct for filter delay (Better way to do this?)
symbols_I = np.real(symbols_I[0:(len(symbols_I))-L+1] / (channel_gain/3))
symbols_Q = np.real(symbols_Q[0:(len(symbols_Q))-L+1] / (channel_gain/3))

# This is how they do filter delay correction #
symbols_I = symbols_I[L-1:(len(symbols_I))-L]
symbols_Q = symbols_Q[L-1:(len(symbols_Q))-L]
###############################################

# Demodulation
buff = np.array([symbols_I,symbols_Q])

demod_bits = demod.symbol_demod(buff, scheme, 1, preamble_length) # gain has to be set to 1

# Error Checking / Decoding
key = np.array([1,1,1,1,0,0,1,0,1])
error = CRC.CRCcheck(demod_bits, key)

if error:
    print("Error!")
else:
    print("No error")

# Data output