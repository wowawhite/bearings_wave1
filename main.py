import os

import numpy as np
import scipy.signal as sg
import scipy.io.wavfile as iowavefile
import os.path as path
import pyaudio ##
''' 
installed via apt: pyaudio-0.9 libalsa? pyaudio-xx? 

'''
import wave
import pywt
import scipy.io.wavfile as audiolib

# os agnostic global relative path calculations
ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '..'))
DB_DIR = path.join(ROOT_DIR, "Storage")
SOUNDFILES_DIR = path.join(ROOT_DIR, "Storage", "Soundfiles")


def infoheader(void):
    infnstring = "Program for continious wavelet calculation. Alpha version."

    print(f', {infnstring}')


# here we nee some sort of mic recording and storing sound sequences in a database as time series data
def RecordAudioSnippet(audioDir, timestamp=None):
    timestamp
    
    audiolib.write()

def openAudiosnippet():
    filename_hardcoded = path.join(SOUNDFILES_DIR, "testfile.wav")
    soundfile = open(filename_hardcoded)
    return soundfile




def testing_pywf(audiofile):
    wavefile = 'path to the wavefile'
    # read the wavefile
    sampling_frequency, signal = iowavefile.read(wavefile)
    #
    scales = (1, len(signal))
    coefficient, frequency = pywt.cwt(signal, scales, 'wavelet_type')


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
def testing_scipy(void):
    # do python magic
    sg.cwt()  # (data, wavelet, widths, dtype=None, **kwargs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    infoheader()
