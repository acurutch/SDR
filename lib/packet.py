import numpy as np
from lib import CRC
import matplotlib.pyplot as plt
from scipy import signal

def preamble_generator(preamble_ones_length): #"a" is the bit array to be modulated


        #The preamble we see in previous labs consists of 20 ones followed by 20 zeros followed by 40 ones and then
        #followed by 20 zeros:

        preamble = np.array([1,0,1,0])
        preamble = np.append(preamble, np.ones(6))                       #np.ones(N) generates a numpy array containing N ones.
        preamble = np.append(preamble, np.zeros(10))   #np.append(a,b) appends array b after array a
##        preamble = np.append(preamble, np.ones(40))
##        preamble = np.append(preamble, np.zeros(20))
        preamble = np.append(preamble, np.ones(preamble_ones_length))
 
        
        return preamble

def packet_generator(data, key, preamble_ones_length):
        
        preamble = preamble_generator(preamble_ones_length)
        packet = np.append(preamble, data)
        packet = CRC.encodeData(packet, key)

        return packet

# Test
data = np.array([1,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,1])
key = np.array([1,1,1,0,0,1,0,1])
preamble_ones_length = 60

packet = packet_generator(data, key, preamble_ones_length)
# print("Packet: ")
# print(packet)

error = CRC.CRCcheck(packet, key)
# print("Error: " + str(error))