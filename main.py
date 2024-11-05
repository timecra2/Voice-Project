import matplotlib.pyplot as plt
import sounddevice
import pydub
import numpy as np
import math

def sinewave(frequency, second, sample_rate=22050,amplitude=1):
    return np.array([ (amplitude * math.sin(frequency * 2 * math.pi *(x / sample_rate))) for x in range(0,second * sample_rate)])

def decibel(x):
    return pow(10,x/10)

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


freq_specturm = [A4*x for x in range(1,10)]
# https://www.youtube.com/watch?v=BvUMfnQucP4
amp_violin = [2.189,1.256,0.459,0.182,0.161,0.016,0.086,0.024,0.011,0.020]

note = sum(sinewave(frequency=freq,second=2,amplitude=amp) for freq, amp in zip(freq_specturm,amp_violin))

sounddevice.play(data=note,samplerate=sample_rate)
sounddevice.wait()

