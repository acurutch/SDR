import numpy as np
from scipy import signal

import pulse_shaping
import matched_filtering
import mode_preconfiguration

import matplotlib.pyplot as plt

preamble = np.array([0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0]).astype(float) # optimal periodic binary code for N = 47 https://ntrs.nasa.gov/citations/19800017860

matched_filter_coef = np.flip(preamble)

data = np.random.randint(2, size=256).astype(float)

testpacket = np.append(preamble,data)

M = 4
pulse_shape = "rrc"
fs = 1e6
# Pulse Shaping
testpacket = pulse_shaping.pulse_shaping(testpacket, M, fs, pulse_shape, 0.9, 8)
matched_filter_coef = pulse_shaping.pulse_shaping(matched_filter_coef, M, fs, pulse_shape, 0.9, 8)

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

####################################
# Add frequency offset - NOTE: Frequency offset of > 1% will result in inability of crosscorrelation operation to detect frame start

# apply a freq offset
fs = 100e6 # assume our sample rate is 1 MHz
fo = 13000 # simulate freq offset
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*(len(testpacket)), Ts) # create time vector
testpacket = testpacket * np.exp(1j*2*np.pi*fo*t) # perform freq shift

"""
NOTES: 
1. Need to apply matched filter first to remove pulse shaping - is this possible? Yes, if you remove the modulation then do phase/freq sync first.
    HOWEVER: This shouldn't actually matter for frame detection, you are correlating with the pulse shaped preamble sequence!
    
2. Frequency offset cannot exceed 1% of carrier frequency or frame detection will not work without phase/freq sync. Q: can you do coarse freq sync on modulated data? Error will just at most be 1 FFT bin (right?)

3. Would be much easier to apply phase/freq sync FIRST and then do frame detection. Not possible with advanced modulation schemes (why?). Actually, with QAM, can just raise to the power of the order of the QAM.
    Could then continuously apply phase/freq sync and store data in buffer, then do frame detection. Will have to store buffer with unsquared data in order to demodulate into symbols. Which is more efficient?

    ALTERNATIVE: run crosscorr continuously to detect preamble for advanced modulation scheme without removing modulation. Works pretty well for most random data I have encountered, max of crosscorr seems to be the starting index, 
    but there are generally other peaks very close by due to pulse shaping.

4. If using BPSK, will have to add differential encoding in order to avoid 180 degree phase problem. (symbol locations flip on IQ plot, no way to tell what is properly 1 or 0)

5. May need to add IQ imbalance correction like they did in Laneman. Or did they correct it in Laneman?

To test:
1. Synchronization on modulated/pulse shaped data - if these methods work then no need to use preamble for synchronization like Laneman code does, can just use preable for frame detection. 
    Can use shorter preamble with much shorter binary code if sync is done before
2. Matched filtering before frame detection
3. Put it all together in synchronization module
"""
scheme="OOK"
L = 8 
total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,len(testpacket),M,fs)
testpacket = matched_filtering.matched_filtering(testpacket, samples_perbit, fs_in, 0.9, L, pulse_shape)

crosscorr = signal.fftconvolve(testpacket,matched_filter_coef)

plt.stem(testpacket)
plt.stem(preamble, 'r')
plt.stem(crosscorr, 'g')
plt.show()

peak_indices, _ = signal.find_peaks(crosscorr)

# peak_indices, _ = signal.find_peaks(crosscorr, height =  peak_threshold, distance = int(0.8*total_samples))
# this finds ALL packets contained in a given buffer

idx = np.array(crosscorr).argmax()

recoveredData = testpacket[idx+1:len(preamble)+len(data)]

## look into correct preamble sequences that minimize side lobes - forgot the name but in iphone pictures