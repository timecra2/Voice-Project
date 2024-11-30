import matplotlib.pyplot as plt
import sounddevice
from pydub import AudioSegment
import numpy as np
import math

#https://www.youtube.com/watch?v=DNKaIe3VTy4

def sinewave(frequency, second, sample_rate,amplitude):
    return np.array([ (amplitude * math.sin(frequency * 2 * math.pi *(x / sample_rate))) for x in range(0,second * sample_rate)])
def decibel(x):
    return pow(10,x/10)




violinAudio = AudioSegment.from_file("violin.mp3")
violinArray = np.array(violinAudio.get_array_of_samples())
violinArray = violinArray.reshape(-1,violinAudio.channels).mean(axis=1)
violinArray = violinArray.astype(np.float32)
violinArray /= np.max(np.abs(violinArray))

audioLength = len(violinArray)/violinAudio.frame_rate

violinDFT = np.fft.fft(violinArray)
violinFreq = np.fft.fftfreq(len(violinArray),1/violinAudio.frame_rate)
violinMag = np.abs(violinDFT)

violinFreqNyq = violinFreq[:len(violinFreq)//2]
violinMagNyq = violinMag[:len(violinMag)//2]
bin_width = violinFreq[1] - violinFreq[0]

plt.plot(violinFreqNyq,violinMagNyq)
plt.show()


freqMagTuples = [(i,j) for i,j in zip(violinFreqNyq,violinMagNyq)]
freqMagDict = {}


for tup in freqMagTuples:
    if round(tup[0],1) in freqMagDict:
        freqMagDict[round(tup[0],1)] += tup[1]
    else :
        freqMagDict[round(tup[0],1)] = tup[1]


freqMagDict = dict(sorted(freqMagDict.items(),key = lambda item : item[1],reverse=True))

amp_violin = []
freq_violin = []



for idx,key in enumerate(freqMagDict):
    if(idx >= 100):
        break
    amp_violin.append(freqMagDict[key])
    freq_violin.append(key)

amp_violin /= amp_violin[0]

note = sum(sinewave(frequency=freq,second=1,amplitude=amp,sample_rate=44100) for freq,amp in zip(freq_violin,amp_violin))

sounddevice.play(note)
sounddevice.wait()

if(True):
    with open("mag.csv",'w') as f:
        magLargest = list(freqMagDict.values())[0]
        for idx,key in enumerate(freqMagDict):
            f.write(f"{key} {freqMagDict[key]/magLargest}\n")



if(False):
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
    freq_specturm = [A4*x for x in range(1,10)]
    # https://www.youtube.com/watch?v=BvUMfnQucP4
    amp_violin = [2.189,1.256,0.459,0.182,0.161,0.016,0.086,0.024,0.011,0.020]

    note = sum(sinewave(frequency=freq,second=2,amplitude=amp) for freq, amp in zip(freq_specturm,amp_violin))

    sounddevice.play(data=note,samplerate=sample_rate)
    sounddevice.wait()

