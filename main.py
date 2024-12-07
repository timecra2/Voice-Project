import matplotlib.pyplot as plt
import sounddevice
from pydub import AudioSegment
import numpy as np
import torch
import os

# references
# https://www.youtube.com/watch?v=DNKaIe3VTy4
# https://www.youtube.com/watch?v=BvUMfnQucP4

def sinewave(frequency, second, sample_rate,amplitude):
    t = np.arange(0,second * sample_rate ) / sample_rate
    return amplitude * np.sin(2 * np.pi*frequency*t)
def decibel(x):
    return pow(10,x/10)

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
        
       # plt.plot(instFreqNyq,instMagNyq)
        #plt.show()  

        self.framerate = instAudio.frame_rate
        self.instDict = Instrument.createFreqMagDict(instFreqNyq,instMagNyq,fineness)

    @staticmethod
    def createFreqMagDict(freqs,mags,fineness):
        bins = np.arange(0,freqs.max() + fineness,fineness)
        binsIndice = np.digitize(freqs,bins)-1
        binsMagnitudes = np.bincount(binsIndice, weights=mags, minlength=len(bins))
        binsMagnitudes /= binsMagnitudes.max()
        binsFreqency = bins[:len(binsMagnitudes)]
        
        return np.column_stack((binsFreqency,binsMagnitudes))
             

    def instMap(self,precision):
        sorted_indices = np.argsort(self.instDict[:,-1])[::-1]
        sorted_dict = self.instDict[sorted_indices]
        return sorted_dict[:precision]

    def writeChart(self,filename):
        with open(filename,'w') as f:
            magMax = max(self.instDict[:,1])
            for idx,tup  in enumerate(self.instDict):
                f.write(f"{tup[0]} {tup[1]/magMax}\n")

    def note(self,second=1,amplitude=1,precision=10):
        return sum(sinewave(frequency=instFreq,
                            second=second,
                            amplitude=instAmp*amplitude,
                            sample_rate=self.framerate) for instFreq,instAmp in self.instMap(precision))


if __name__ == "__main__":
    source = input("Write file to read as instrument: ")
    gap = float(input("Write finess of sound discerning such as 0.01(hz): "))
    inst = Instrument(source,fineness=gap)
    note = inst.note(second=5,amplitude=0.1,precision=100)
    normNote = (note*32767).astype(np.int16)
    audioFile = AudioSegment(
        normNote.tobytes(),
        frame_rate=inst.framerate,
        sample_width = normNote.dtype.itemsize,
        channels=1
    )

    sourceName = os.path.splitext(os.path.basename(source))[0]
    output_path = f"{sourceName}_modelled.mp3"
    audioFile.export(output_path, format="mp3")
    print(f"MP3 file saved to {output_path}")

    
    


