import matplotlib.pyplot as plt
import sounddevice
import pydub
import numpy as np
import math

def sinewave(frequency, second, sample_rate=22050,amplitude=1):
    return np.array([ amplitude * math.sin(frequency * 2 * math.pi *(x / sample_rate)) for x in range(0,second * sample_rate)])

sample_rate = 22050

A4 = 440
B4 = A4 * pow(2,2/12)
C5 = A4 * pow(2,3/12)
D5 = A4 * pow(2,5/12)
E5 = A4 * pow(2,7/12)
F5 = A4 * pow(2,8/12)
G5 = A4 * pow(2,10/12)
A6 = A4 * pow(2,12/12)
B6 = A4 * pow(2,14/12)
C6 = A4 * pow(2,15/12)


movingman = pydub.AudioSegment.from_file("movingman.mp3")


note = np.concatenate([sinewave(C5,1),
                       sinewave(D5,1),
                       sinewave(E5,1),
                       sinewave(F5,1),
                       sinewave(G5,1),
                       sinewave(A6,1),
                       sinewave(B6,1),
                       sinewave(C6,1)],axis=None)

sounddevice.play(data=note,samplerate=sample_rate)
sounddevice.wait()

