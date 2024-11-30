import matplotlib.pyplot as plt
import sounddevice
from pydub import AudioSegment
import numpy as np
import math

# references
# https://www.youtube.com/watch?v=DNKaIe3VTy4
# https://www.youtube.com/watch?v=BvUMfnQucP4


def sinewave(frequency, second, sample_rate,amplitude):
    return np.array([ (amplitude * math.sin(frequency * 2 * math.pi *(x / sample_rate))) for x in range(0,second * sample_rate)])
def decibel(x):
    return pow(10,x/10)


#(amplitude,t) -> (amplitude,f)
class Instrument:
    def __init__(self,filename):
        instAudio = AudioSegment.from_file(filename)   
        instArray = np.array(instAudio.get_array_of_samples())
        instArray = instArray.reshape(-1,instAudio.channels).mean(axis=1)
        instArray /= np.max(np.abs(instArray))

        instDFT = np.fft.fft(instArray)
        instFreq = np.fft.fftfreq(len(instArray),1/instAudio.frame_rate)
        instMag =np.abs(instDFT)

        instFreqNyq = instFreq[:len(instFreq)//2]
        instMagNyq = instMag[:len(instMag)//2]
        instMagNyq /= instMagNyq.max()
        
        plt.plot(instFreqNyq,instMagNyq)
        plt.show()

        freqMagTuples = [(i,j) for i,j in zip(instFreqNyq,instMagNyq)]
        freqMagDict = {}
        for tup in freqMagTuples:
            if round(tup[0],1) in freqMagDict:
                freqMagDict[round(tup[0],1)] += tup[1]
            else :
                freqMagDict[round(tup[0],1)] = tup[1]
        freqMagDict = dict(sorted(freqMagDict.items(),key = lambda item : item[1],reverse=True))

        self.framerate = instAudio.frame_rate
        self.instDict = freqMagDict

    def instMap(self,precision):
        instMap = []
        for idx,key in enumerate(self.instDict):
            if(idx >= precision):
                break
            instMap.append((key,self.instDict[key]))
        return instMap

    def writeChart(self,filename):
        with open(filename,'w') as f:
            magMax = max(self.instDict.values())
            for idx,key  in enumerate(self.instDict):
                f.write(f"{key} {self.instDict[key]/magMax}\n")

    def note(self,freq=1,second=1,amplitude=1,precision=10):
        return sum(sinewave(frequency=instFreq,
                            second=second,
                            amplitude=instAmp*amplitude,
                            sample_rate=self.framerate) for instFreq,instAmp in self.instMap(precision))







acord = Instrument("inst/acord.mp3")



sounddevice.play(acord.note(second=5,amplitude=0.1,precision=100))
sounddevice.wait()
acord.writeChart("acord.csv")








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

    amp_violin = [2.189,1.256,0.459,0.182,0.161,0.016,0.086,0.024,0.011,0.020]

    note = sum(sinewave(frequency=freq,second=2,amplitude=amp) for freq, amp in zip(freq_specturm,amp_violin))

    sounddevice.play(data=note,samplerate=sample_rate)
    sounddevice.wait()

