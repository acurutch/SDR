import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

###################################3
# Pulse Shaping

num_symbols = 100
sps = 8
bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's
pulse_train = np.array([])

for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    pulse_train = np.concatenate((pulse_train, pulse)) # add the 8 samples to the signal

plt.stem(pulse_train)
plt.stem(bits, 'ro')
plt.show()

# Create our raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(-51, 52) # remember it's not inclusive of final number
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# Filter our signal, in order to apply the pulse shaping
samples = np.convolve(pulse_train, h)

# Plot
plt.stem(np.real(samples))
plt.stem(np.imag(samples), 'ro')
plt.show()

#################################
# Add fractional delay

# Create and apply fractional delay filter
delay = 0.4 # fractional delay, in samples
N = 21 # number of taps
n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
h = np.sinc(n - delay) # calc filter taps
h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
samples = np.convolve(samples, h) # apply filter

####################################
# Add frequency offset

# apply a freq offset
fs = 1e6 # assume our sample rate is 1 MHz
fo = 13000 # simulate freq offset
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(1j*2*np.pi*fo*t) # perform freq shift

# Plot
plt.stem(np.real(samples))
plt.stem(np.imag(samples), 'ro')
plt.show()


#####################################
# May need to add IQ imbalance correction like they did in laneman.

#####################################
# Muller and Muller Clock recovery

samples_interpolated = signal.resample_poly(samples, 16, 1)

mu = 0 # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=complex)
out_rail = np.zeros(len(samples) + 10, dtype=complex) # stores values, each iteration we need the previous 2 values plus current value
i_in = 0 # input samples index
i_out = 2 # output index (let first two outputs be 0)

while i_out < len(samples) and i_in+16 < len(samples):
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
samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on

# Plot
plt.stem(np.real(samples))
plt.stem(np.imag(samples), 'ro')
plt.show()

#####################################
# Interpolate samples

# plt.figure("Pulse Train")
# plt.stem(pulse_train)
# plt.figure("Clock recovery")
# plt.stem(np.real(out), 'ro')
# plt.stem(np.imag(out), 'bo')

# plt.show()

########################################
# Coarse Frequency Detection and Correction
fft_samples = samples**2

psd = np.fft.fftshift(np.abs(np.fft.fft(fft_samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))

max_freq = f[np.argmax(psd)]
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0) # not perfectly synchronized -- coarse sync

# Plot
plt.stem(np.real(samples))
plt.stem(np.imag(samples), 'ro')
plt.show()

######################################
# Costas Loop Fine Freq Correction

N = len(samples)
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=complex)
freq_log = []
for i in range(N):
    out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
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

samples = out

plt.stem(samples)
plt.show()

fft_samples = samples**2

# Coarse Frequency Detection and Correction
psd = np.fft.fftshift(np.abs(np.fft.fft(fft_samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))


plt.plot(f, psd)
plt.show()

####################################
# Add Frame synchronization using Willard 13 sequence (unity sidelobes)