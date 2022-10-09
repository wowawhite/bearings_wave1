import numpy as np
import scipy.signal as sg
import os.path as path
import pyaudio
import wave

# os agnostic global relative path calculations
ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '..'))
DB_DIR = path.join(ROOT_DIR,"Storage")
SOUNDFILES_DIR = path.join(ROOT_DIR,"Storage","Soundfiles")

def infoheader(void):
    infnstring = "Program for continious wavelet calculation. Alpha version."

    print(f', {infnstring}')

# here we nee some sort of mic recording and storing sound sequences in a database as time series data


def readfile():
    filename_hardcoded = path.join(SOUNDFILES_DIR, "testfile.wav")

    soundfile = open(filename_hardcoded)
    return soundfile


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
def wavelet_stuff(void):
    # do python magic
    sg.cwt()  # (data, wavelet, widths, dtype=None, **kwargs)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    infoheader()


