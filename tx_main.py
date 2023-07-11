import pulse_shaping

import mode_preconfiguration

import packet
import CRC

import symbol_mod_QAM as mod

import numpy as np

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

# Data intake
data = np.random.randint(0,2,size=256) 

dataLen = len(data)

known_preamble_bits = packet.preamble_generator(preamble_ones_length)
preamble_length = len(known_preamble_bits)

# Coding and Packet formation
key = np.array([1,1,1,1,0,0,1,0,1])

dataPacket = packet.packet_generator(data, key, preamble_ones_length)

packetLen = len(dataPacket)
remainderLen = len(key)-1

# Modulation
baseband_symbols = mod.symbol_mod(dataPacket, scheme, preamble_length)

data_symbols = mod.symbol_mod(data, scheme, 0)

# Pulse Shaping
baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, 0.9, 8)

# Transmission (to socket)