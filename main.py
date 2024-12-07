import matplotlib.pyplot as plt
import sounddevice
from pydub import AudioSegment
import numpy as np
import math
import torch


# references
# https://www.youtube.com/watch?v=DNKaIe3VTy4
# https://www.youtube.com/watch?v=BvUMfnQucP4


def sinewave(frequency, second, sample_rate,amplitude):
    t = np.arange(0,second * sample_rate ) / sample_rate
    return amplitude * np.sin(2 * np.pi*frequency*t)
def decibel(x):
    return pow(10,x/10)

SEMITONE = (2 ** (1/12))
A0 = 27.5

#(amplitude,t) -> (amplitude,f)
class Instrument:
    def __init__(self,filename,fineness):
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

        self.framerate = instAudio.frame_rate
        self.instDict = Instrument.createFreqMagDict(instFreqNyq,instMagNyq,fineness)
        self.fineness = fineness

    @staticmethod
    def createFreqMagDict(freqs,mags,fineness):

        nBins = int(math.log(freqs.max()/A0,fineness))
        bins = [A0 * fineness**(i) for i in range(0,nBins)]
        indices = np.digitize(x=freqs,bins=bins)   
        binsMagnitudes = np.zeros(nBins) 
        np.add.at(binsMagnitudes,indices-1,mags)
        binsMagnitudes /= binsMagnitudes.max()
        return np.column_stack((bins,binsMagnitudes))
             

    def instMap(self,precision):
        sorted_indices = np.argsort(self.instDict[:,-1])[::-1]
        sorted_dict = self.instDict[sorted_indices]
        return sorted_dict[:precision]

    def writeChart(self,filename):
        with open(filename,'w') as f:
            magMax = max(self.instDict[:,1])
            for idx,tup  in enumerate(self.instDict):
                f.write(f"{tup[0]} {tup[1]/magMax}\n")

    def note(self,second=1,amplitude=0.1,precision=10):
        return sum(sinewave(frequency=instFreq,
                            second=second,
                            amplitude=instAmp*amplitude,
                            sample_rate=self.framerate) for instFreq,instAmp in self.instMap(precision))
    
    def plot(self):
        nBins = int(math.log(22050/A0,self.fineness))
        plt.plot(self.instDict[:,0],self.instDict[:,1])
        plt.scatter([A0 * self.fineness ** i for i in range(0,nBins)],np.zeros(nBins),s=3,color=[1,0,0])
        plt.show()

        







violin = Instrument("inst/violin.mp3",fineness=SEMITONE)
violin.writeChart("violin.csv")
violin.plot()


#sounddevice.play(violin.note(second=5,amplitude=1,precision=2))
#sounddevice.wait()






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

