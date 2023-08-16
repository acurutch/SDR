import pulse_shaping

import mode_preconfiguration

import packet
import CRC

import symbol_mod_QAM as mod
import symbol_demod_QAM as demod
import matched_filtering

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Declare Variables
fs = 2.4e9    
Ts = 1.0 / fs          # sampling period in seconds
f0 = 0.0               # homodyne (0 HZ IF)
M = 8 # oversampling factor
T = M*Ts               # symbol period in seconds
Rs = 1/T               # symbol rate
segment_size = 16    # Might need to alter this to reflect actual length of packet
R = 1               # Packet Ratio: number of segments contained in our larger OOK packet 
N = R*segment_size     # OOK Packet Length (equals R* segment_size)
L = 8                  # pulse shaping filter length
# Is this the ideal length
pulse_shape = 'rrc' # type of pulse shaping, also "rect"
scheme = 'BPSK'    # Modulation scheme 'OOK', 'QPSK', or 'QAM'
preamble_ones_length = 0
alpha = 0.5 #roll-off factor of the RRC matched-filter

# # Data intake
# data = np.random.randint(0,2,size=256) 

# dataLen = len(data)

# known_preamble_bits = packet.preamble_generator(preamble_ones_length)
# preamble_length = len(known_preamble_bits)
# matched_filter_coef = np.flip(known_preamble_bits)

# # Coding and Packet formation
# key = np.array([1,1,1,1,0,0,1,0,1])

# dataPacket = packet.packet_generator(data, key, preamble_ones_length)

# packetLen = len(dataPacket)
# remainderLen = len(key)-1

# # Modulation
# baseband_symbols = mod.symbol_mod(dataPacket, scheme, preamble_length)

# data_symbols = mod.symbol_mod(data, scheme, 0)

# plt.stem(np.real(baseband_symbols))
# plt.stem(np.imag(baseband_symbols), 'ro')
# plt.show()


# # Pulse Shaping
# baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, 0.9, 8)
# matched_filter_coef = pulse_shaping.pulse_shaping(matched_filter_coef, M, fs, pulse_shape, 0.9, 8)

# # Transmission (to socket)
# testpacket = baseband

# # Plot
# plt.stem(np.real(testpacket))
# plt.stem(np.imag(testpacket), 'ro')
# plt.show()

# Old data generation method ^^^
##################################

###################################
# Pulse Shaping

num_symbols = 100
sps = 8
# bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's

# preamble = np.array([0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0]).astype(float) # optimal periodic binary code for N = 47 https://ntrs.nasa.gov/citations/19800017860
preamble = np.array([0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,1,0]).astype(float) # optimal periodic binary code for N = 63 https://ntrs.nasa.gov/citations/19800017860

matched_filter_coef = np.flip(preamble)

data = np.random.randint(2, size=256).astype(float)

bits = np.append(preamble,data)

pulse_train = np.array([])

for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    pulse_train = np.concatenate((pulse_train, pulse)) # add the 8 samples to the signal

# plt.stem(pulse_train)
# plt.stem(bits, 'ro')
# plt.show()

# y = signal.upfirdn([1],bits,M)

# # Create our raised-cosine filter
# num_taps = 101
# beta = 0.35
# Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
# t = np.arange(-51, 52) # remember it's not inclusive of final number
# h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# # Filter our signal, in order to apply the pulse shaping
# samples = np.convolve(pulse_train, h)

# # Plot
# plt.stem(np.real(samples))
# plt.stem(np.imag(samples), 'ro')
# plt.show()

# plt.stem(samples, 'bo')
symbols_I = pulse_shaping.pulse_shaping(pulse_train, sps, fs, pulse_shape, alpha, L)
#plt.stem(symbols_I, 'ro')
# symbols_I = matched_filtering.matched_filtering(symbols_I, sps, fs, alpha, L, pulse_shape)
# plt.stem(symbols_I, 'mo')
# plt.stem(signal.upfirdn([1],pulse_train,1, sps), 'ko')
# plt.show()

testpacket = symbols_I

# plt.stem(samples, 'mo')
plt.stem(symbols_I, 'ko')
plt.title("After pulse shaping")
plt.show()

###########################
# Generate Noise

# Parameters for AWGN
mean = 0      # Mean of the Gaussian distribution (usually 0 for AWGN)
std_dev = 0.1   # Standard deviation of the Gaussian distribution

# Number of samples
num_samples = len(testpacket)

# Generate AWGN samples
awgn_samples = np.random.normal(mean, std_dev, num_samples)

testpacket += awgn_samples

# # Plot
# plt.stem(np.real(testpacket))
# plt.stem(np.imag(testpacket), 'ro')
# plt.show()

#################################
# Add fractional delay

# Create and apply fractional delay filter
delay = 0.4 # fractional delay, in samples
N = 21 # number of taps
n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
h = np.sinc(n - delay) # calc filter taps
h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
testpacket = np.convolve(testpacket, h) # apply filter

###################################
# Add frequency offset - NOTE: Frequency offset of > 1% will result in inability of crosscorrelation operation to detect frame start

# apply a freq offset
fs = 2.45e9 # assume our sample rate is 1 MHz
fo = 61250 # simulate freq offset
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*(len(testpacket)), Ts) # create time vector
testpacket = testpacket * np.exp(1j*2*np.pi*fo*t) # perform freq shift

#####################################
# Matched Filtering
# symbols_I = matched_filtering.matched_filtering(testpacket, sps, fs, alpha, L, pulse_shape)
# testpacket = symbols_I

# testpacket = signal.upfirdn([1],testpacket,sps)

#####################################
# Muller and Muller Clock recovery
# Only works for BPSK, not for other modulation schemes
# testpacket = symbols_I
samples_interpolated = signal.resample_poly(testpacket, 16, 1)

# Plot
plt.stem(np.real(samples_interpolated), 'bo')
plt.stem(np.real(testpacket), 'ro')
plt.title("After non-idealities")
plt.show()

mu = 0 # initial estimate of phase of sample
out = np.zeros(len(testpacket) + 10, dtype=complex)
out_rail = np.zeros(len(testpacket) + 10, dtype=complex) # stores values, each iteration we need the previous 2 values plus current value
i_in = 0 # input samples index
i_out = 2 # output index (let first two outputs be 0)
sps=8

while i_out < len(testpacket) and i_in+16 < len(testpacket):
    out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
    y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y - x)
    mu += sps + 0.3*mm_val
    i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu) # remove the integer part of mu
    i_out += 1 # increment output index
out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
testpacket = out # only include this line if you want to connect this code snippet with the Costas Loop later on



# Plot
plt.stem(signal.upfirdn([1],pulse_train,1, sps), 'ko')
plt.stem(np.real(testpacket), 'ro')
plt.stem(np.imag(testpacket), 'mo')
# plt.stem(np.imag(testpacket), 'ro')
plt.title("After clock recovery")
plt.show()

########################################
# Coarse Frequency Detection and Correction
fft_samples = testpacket**2

psd = np.fft.fftshift(np.abs(np.fft.fft(fft_samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))

max_freq = f[np.argmax(psd)]
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(testpacket), Ts) # create time vector
testpacket = testpacket * np.exp(-1j*2*np.pi*max_freq*t/2.0) # some error here with length of time vector.

plt.plot(f, psd)
plt.title("Frequency offset before correction")
plt.show()

# Plot
plt.stem(signal.upfirdn([1],pulse_train,1, sps), 'ko')
plt.stem(np.real(testpacket))
plt.stem(np.imag(testpacket), 'ro')
plt.title("After coarse frequency correction")
plt.show()

######################################
# Costas Loop Fine Freq Correction

N = len(testpacket)
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=complex)
freq_log = []
for i in range(N):
    out[i] = testpacket[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq * fs / (2*np.pi)) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)

    # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2*np.pi:
        phase -= 2*np.pi
    while phase < 0:
        phase += 2*np.pi

# Plot freq over time to see how long it takes to hit the right offset
# plt.plot(freq_log,'.-')
# plt.show()

testpacket = out

# Plot
plt.stem(signal.upfirdn([1],pulse_train,1, sps), 'ko')
plt.stem(np.real(testpacket))
plt.stem(np.imag(testpacket), 'ro')
plt.title("After Costas Loop fine frequency correction")
plt.show()

# Testing frequency correction
fft_samples = testpacket**2
psd = np.fft.fftshift(np.abs(np.fft.fft(fft_samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))

plt.plot(f, psd)
plt.title("Frequency offset after correction")
plt.show()

#####################################
# Frame Sync
scale = np.mean(np.abs(testpacket))
out = np.array([])

for symbol in testpacket:
    symbol = (symbol + scale)/2*scale
    out = np.append(out, symbol)

crosscorr = signal.fftconvolve(out,matched_filter_coef)

plt.stem(np.real(testpacket))
plt.stem(np.imag(testpacket), 'ro')
plt.stem(preamble, 'r')
plt.stem(crosscorr, 'g')
plt.title("Frame Synchronization: Crosscorrelation")
plt.show()

peak_indices, _ = signal.find_peaks(crosscorr)

# peak_indices, _ = signal.find_peaks(crosscorr, height =  peak_threshold, distance = int(0.8*total_samples))
# this finds ALL packets contained in a given buffer

idx = np.array(crosscorr).argmax()

recoveredPayload = testpacket[idx-len(preamble)+1:idx+len(data)+1]
recoveredData = recoveredPayload[len(preamble):]

# Plot
plt.stem(np.real(recoveredPayload))
plt.stem(np.imag(recoveredPayload), 'ro')
plt.stem(signal.upfirdn([1],pulse_train,1, sps), 'ko')
plt.title("Data recovery")
plt.show()

# ####################################
# # Matched Filtering and Synchronization
# scheme="OOK"
# L = 8 
# total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,len(testpacket),M,fs)
# testpacket = matched_filtering.matched_filtering(testpacket, samples_perbit, fs_in, 0.9, L, pulse_shape)

# # Plot
# plt.stem(np.real(testpacket))
# plt.stem(np.imag(testpacket), 'ro')
# plt.show()

demod_bits = demod.symbol_demod(recoveredData, scheme, 1, len(preamble)) # gain has to be set to 1

# Plot
plt.stem(np.real(demod_bits))
plt.stem(0.5*np.real(data), 'ro')
plt.title("Demodulated bit comparison")
plt.show()

####################################
# To do 

# 1. For packet detection, add threshold slightly less than number of ones in preamble
# 2. Still need to look into equalizing samples on clock recovery-- why won't clock recovery work with pulse train?
# 3. Look into equalizing channel gain
# 4. Look into IQ imbalance correction