#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
import scipy.io.wavfile as wavefile
import os.path as path
import os
import pywt
import wave
import time
from pathlib import Path
from Wavelet2Go import Wavelet2Go


''' 
installed via apt: pyaudio-0.9 libalsa? pyaudio-xx? 
used to read mic audio input from windows and linux
'''
def infoheader():
    SW_VERSION = 0.01  # increment sw version
    infostring = "Program for continious wavelet calculation. Software version:"
    print('System info: ', os.sys.platform)
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'))
    print(f'{infostring} {SW_VERSION}')
def foldermagic():
    # check if windows or linux os. OSX is not supported
    if os.sys.platform == 'win32':
        system_is_windows = True
    else:
        system_is_windows = False
    # os agnostic global relative path calculations
    ROOT_DIR = path.realpath(path.join(path.dirname(__file__)))
    DB_DIR = path.join(ROOT_DIR, "Storage")
    SOUNDFILES_DIR = path.join(ROOT_DIR, "Storage", "Soundfiles")

    # create folder structure if not available
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    Path(SOUNDFILES_DIR).mkdir(parents=True, exist_ok=True)
    print('ROOTDIR: ', {ROOT_DIR})
    print('DB_DIR: ', {DB_DIR})
    print('SOUNDFILES_DIR: ', {SOUNDFILES_DIR})
    return ROOT_DIR, DB_DIR, SOUNDFILES_DIR



# here we nee some sort of mic recording and storing sound sequences in a database as time series data
def RecordAudioSnippet(audioDir, timestamp=None):
    # TODO: create wav files from mic recording
    # wavefile.write()
    return

def write_to_database(data_container):
    ''' TODO foo to add data container to a time series database
    plan:
    1. create influxDB in local container
    2.
    '''
    return
def openAudiosnippet(folderstruct, audio_filename):

    ROOT_DIR, DB_DIR, SOUNDFILES_DIR = folderstruct

    soundfilestring = path.join(SOUNDFILES_DIR, audio_filename)
    print(f'Reading file:{soundfilestring}')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
    samplerate, data = wavefile.read(soundfilestring)  # return is samplerate, stereo 32 bit float 2d array
    t = np.arange(len(data)) / float(samplerate)  # Retrieving Time
    l_audio = len(data.shape)
    if l_audio == 2:
        signal = data.sum(axis=1) / 2
    N = data.shape[0]
    print("Audio Channels", l_audio)
    secs = N / float(samplerate)
    print("Soundfile length", secs)
    Ts = 1.0 / samplerate  # sampling interval in time
    print("Timestep between samples Ts", Ts)
    # debug stuff
    '''
    print("Start time for every sample point-> ", t)
    print('Time Data points', np.size(t))
    print('Data points', np.size(data))
    print('Data shape', np.shape(data))
    print("samplerate -> ", float(samplerate))
    print("Datatype -> ", type(t), type(data))
    # print("data -> ", data)
    print("data length sample points -> ", len(data))
    '''
    maxval = np.amax(data)
    data_normalized = data /maxval
    return samplerate, data_normalized

def calculate_fourer(folderstructure, audiocontainer):
    '''
    fft copypasted from here
    https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
    '''
    samplerate, data_normalized = audiocontainer
    '''
    #b = data_normalized.T[0, 0:data_normalized.size]  # transponsing and slicing to mono
    #b = np.array(data_normalized.T[:, 0])
    #b = [(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    #signal = fft(b)  # calculate fourier transform (complex numbers list)
    #d = np.size(signal)//2  # you only need half of the fft list (real signal symmetry)
  
    print('fft Data shape', np.shape(b))
    print('fft Data shape', np.shape(c))
    print('fft Data shape', np.shape(d))
    #print("Datatype -> ", type(b), type(c), type(d))
    print('d',d, type(d))
    half_data =c[0:(d-1)]
    plt.plot(abs(c), 'r')
    plt.show()
    '''

    l_audio = len(data_normalized.shape)
    # shape stereo to mono
    if l_audio == 2:
        signal = data_normalized.sum(axis=1) / 2  #summing elements from axis 1
    else:
        signal = data_normalized
    N = signal.shape[0]

    print("singal shape", np.shape(signal))

    secs = N / float(samplerate)

    Ts = 1.0/samplerate # sampling interval in time

    t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray

    FFT = scipy.fftpack.fft(signal)
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    fft_freqs_side = np.array(freqs_side)

    # plotting
    plt.subplot(311)
    p1 = plt.plot(t, signal, "g")  # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    p2 = plt.plot(freqs, FFT, "r")  # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count dbl-sided')
    plt.subplot(313)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")  # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.show()
    return

def testing_pywf_1(folderstructure, audiocontainer):
    '''tut from here:
    https://python.plainenglish.io/discrete-wavelet-transform-analysis-on-audio-signals-using-python-94744a418601
    advanced wavelet foos
    http://www.neurotec.uni-bremen.de/drupal/node/46
    '''
    ROOT_DIR, DB_DIR, SOUNDFILES_DIR = folderstruct
    samplerate, raw_data = audiocontainer  # still stereo data

    ''' 
    
    '''
    cA, cD = pywt.dwt(raw_data, 'bior6.8', 'per') #DWT
    y = pywt.idwt(cA, cD,  'bior6.8', 'per') #IDWT

    sample_R_file = path.join(SOUNDFILES_DIR, 'sample_R.wav')
    sample_cD_file = path.join(SOUNDFILES_DIR, 'sample_cD.wav')
    sample_cA_file = path.join(SOUNDFILES_DIR, 'sample_cA.wav')

    wavefile.write(sample_R_file, samplerate, y);
    wavefile.write(sample_cD_file, samplerate, cD);
    wavefile.write(sample_cA_file, samplerate, cA);

    print('Audiofiles created in folder: ', {SOUNDFILES_DIR})

    return True

def testing_pywf_2(folderstruct, container_audio_data):
    # plot tut from https://medium.com/@shouke.wei/plot-approximations-of-wavelet-and-scaling-functions-in-python-dabd31452bdf
    wavelet = pywt.Wavelet('db5')
    [phi, psi, x] = wavelet.wavefun(level=1)
    plt.plot(x, psi)
    plt.show()

def testing_pywf_3(folderstruct, container_audio_data):
    wavlist = pywt.wavelist(kind="continuous")
    print(wavlist)  # --> ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']
    wavelet_name: str = "cmor1.5-1.0"

    # "linked" to how many peaks and
    # troughs the wavelet will have
    scale: float = 10

    # Invoking the complex morlet wavelet object
    wav = pywt.ContinuousWavelet("cmor1.5-1.0")

    # Integrate psi wavelet function from -Inf to x
    # using the rectangle integration method.
    int_psi, x = pywt.integrate_wavelet(wav, precision=10)
    int_psi /= np.abs(int_psi).max()
    wav_filter: np.ndarray = int_psi[::-1]

    nt: int = len(wav_filter)
    t: np.ndarray = np.linspace(-nt // 2, nt // 2, nt)
    plt.plot(t, wav_filter.real)
    plt.plot(t, wav_filter.imag)
    plt.ylim([-1, 1])
    plt.legend(["real", "imaginary"], loc="upper left")
    plt.xlabel("time (samples)")
    plt.title("filter " + wavelet_name)
    plt.show()


    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ' global variables are always a bad idea'
    infoheader()  # print basic software info
    folderstruct = foldermagic()  # calculate paths
    container_audio_data = openAudiosnippet(folderstruct, 'piano2.wav')

    #testing_pywf_1(folderstruct, container_audio_data)
    #testing_pywf_2(folderstruct, container_audio_data)
    testing_pywf_3(folderstruct, container_audio_data)
    #calculate_fourer(folderstruct, container_audio_data)

