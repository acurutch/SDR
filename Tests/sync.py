import pyfftw
import numpy as np
import matplotlib.pyplot as plt

def coarse_freq_sync(data, payload_start, samples_perbit, fs_in):
        
        #Estimate the frequency offset

        num_of_bits_fft = 180
        segments_of_data_for_fft = data[int(payload_start-num_of_bits_fft*samples_perbit):payload_start]
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
        coarse_frequency = (peak_position-len(spectrum)/2) / len(spectrum) * fs_in

        return coarse_frequency

def fine_freq_sync(data, payload_start, preamble_length, samples_perbit, coarse_frequency, fs_in):

    num_of_bits_fft = preamble_length-20
    preamble = data[int(payload_start-num_of_bits_fft*samples_perbit):payload_start]

    original_fft_point = np.power(2,20)

    center_index = int(coarse_frequency*original_fft_point/fs_in)
    A = preamble
    x_max = - 1000
    peak_position = 0

    for k in range(center_index-4, center_index + 5):
            x = abs(np.sum(A*np.exp(-1j*2*np.pi*k*np.arange(len(preamble))/original_fft_point)))
            if(x>x_max):
                    peak_position = k
                    x_max = x

    #Calculate the fine frequency
    carrier_frequency = fs_in*peak_position/original_fft_point

    return carrier_frequency

def phase_sync(data, payload_start, samples_perbit, preamble_length, scheme, carrier_frequency, N, fs_in):
    if(scheme == "QPSK"):
        N = int(N/2)

    if(scheme == "QAM"):
            N = int(N/4)

    Ts_in = 1/fs_in

    ### test to make sure this outputs the correct payload ###
    # payload_before_correction = data[payload_start:(payload_start + N*samples_perbit)]
    
    # ones_length = preamble_length - 20
    
    # payload_and_ones = data[int(payload_start-ones_length):]
    
    # k = np.arange(len(payload_and_ones))

    k = np.arange(len(data))
    
    Digital_LO = np.exp(-1j*2*np.pi*carrier_frequency*(k*Ts_in))

    #Correct the frequency and then extract & correct the phase

    #First, correct the frequency offset from packet_data
    #Hint: You may find np.multiply() useful
    #packet_data_freq_corrected = 
    # packet_data_freq_corrected = np.multiply(payload_and_ones,Digital_LO)
    packet_data_freq_corrected = np.multiply(data,Digital_LO)
    plt.plot(packet_data_freq_corrected)

    #remove the BB voltage offset at the payload due to non-idealities
    # packet_data_freq_corrected = packet_data_freq_corrected - np.mean(packet_data_freq_corrected[payload_start:])
    # packet_data_freq_corrected = packet_data_freq_corrected - np.mean(packet_data_freq_corrected)

    #Extract the preamble only from the corrected packet (preamble + payload)
    preamble = packet_data_freq_corrected[0:int(preamble_length*samples_perbit)]

    #Extract carrier phase offset using "preamble" above
    #Hint: You may find np.angle() useful
    #angles =
    angles = np.angle(np.abs(preamble))

    #Averaging for better estimate
    phase_estimated = np.mean(angles)

    #Correct the carrier phase offset in "packet_data_freq_corrected" to obtain signal samples with both frequency and phase offsets corrected
    #Hint: You may find np.multiply() helpful and you may want to construct a complex exponential using "phase_estimated"
    #phase_corrected_packet = 
    phase_corrected_packet = np.multiply(packet_data_freq_corrected,np.exp(-1j*phase_estimated))    
    plt.plot(phase_corrected_packet)

    # payload_corrected = phase_corrected_packet[ones_length*samples_perbit:]

    # baseband_signal_I_new = np.real(payload_corrected)
    # baseband_signal_Q_new = np.imag(payload_corrected)   
    
    
    baseband_signal_I_new = np.real(phase_corrected_packet)
    baseband_signal_Q_new = np.imag(phase_corrected_packet)     

    return baseband_signal_I_new, baseband_signal_Q_new

def frame_sync():
    return