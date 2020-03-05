import wave as we
import scipy
import numpy as np
from scipy.signal import *
import matplotlib.pyplot as plt

# 读取WAV文件数据
def wavread(path):
    wavfile = we.open(path,'rb')
    params = wavfile.getparams()
    framerate,framewave = params[2],params[3]
    datawav = wavfile.readframes(framewave)
    wavfile.close()
    audio = np.fromstring(datawav,dtype=np.short)
    audio.shape = -1,1
    audio = audio.T
    time = np.arange(0,framewave)*(1.0/framerate)
    return audio,time,framerate

# 处理WAV文件数据
def fft_filter(signal):
    sampling_rate = 44100
    fft_size = 2200
    win = hann(2200)
    f1 = firwin(101,0.5,window=('kaiser',8),pass_zero = False)
    freqs = np.linspace(0,int(sampling_rate/2),int(fft_size/2+1))
    xs = signal[5800:8000] * win
    xc = np.convolve(xs,f1,mode='same')
    xf = np.fft.rfft(xc)/fft_size
    xn = np.fft.rfft(xs)/fft_size
    xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    xfn = 20*np.log10(np.clip(np.abs(xn), 1e-20, 1e100))
    return xfp,freqs,xc,xs,xfn

# 绘图
def main():
    path =  'eva.wav' #input("The path is:")
    audio,time,framerate = wavread(path)
    audio_left = audio[0]

    y1,x1,z1,k1,n1 = fft_filter(audio_left)

    plt.subplot(421)
    plt.plot(time[5800:8000],k1,color = 'green')

    plt.subplot(422)
    plt.plot(time[5800:8000],z1,color='green')
    plt.title("Filter Output")
    plt.xlabel("Frequency(Hz)")

    plt.subplot(423)
    plt.plot(x1,n1,color='green')
    plt.title("Flute Spectrum Before Filter")
    plt.ylabel("Magnitude(dB)")
    plt.xlabel("Frequency(Hz)")

    plt.subplot(424)
    plt.plot(x1,y1,color='green')
    plt.title("Flute Spectrum")
    plt.ylabel("Magnitude(dB)")
    plt.xlabel("Frequency(Hz)")

    plt.show()

main()