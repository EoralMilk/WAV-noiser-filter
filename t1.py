import wave as we
import numpy as np
import matplotlib.pyplot as plt


def wavread(path):
    wavfile = we.open(path,"rb")                        #打开WAV文件
    params = wavfile.getparams()                        #获取WAV头文件中的参数
    nchannels, sampwidth, framerate, nframes= params[:4]            #获取声道数，量化位数，采样率和帧数
    datawav = wavfile.readframes(nframes)               #读取每一帧数据
    wavfile.close()                                     #关闭文件
    datause = np.fromstring(datawav,dtype = np.short)   #将String格式转换成int型数据
    datause.shape = -1,nchannels                        #将横向矩阵转为纵向矩阵，注意声道数
    datause = datause.T	                                #矩阵转置
    time = np.arange(0, nframes) * (1.0/framerate)      #返回一个时间长度的list
    return datause,time,nchannels, sampwidth, framerate, nframes

def getfreq(datause,nframes,framerate,start):   # 时域转频域
    N = nframes                                 # 采样点数，修改采样点数进行不同位置和长度的音频波形分析
    df = framerate/(N-1)                        # 分辨率
    freq = [df*n for n in range(0,N)]           # N个元素
    tempdata = datause[0][start:start+N]
    freqdata = np.fft.fft(tempdata)*2/N
    d=int(len(freqdata)/2)                      # 常规显示采样频率一半的频谱
    return freq,d,freqdata

def freq2time(nframes,freqdata):    # 频域转时域
    fdata = freqdata*nframes
    filter_sig = np.fft.ifft(fdata).real
    data = filter_sig.astype(np.short)
    return data


def filter(freq,d,freqdata): # 滤波器
    while d:
        d-=1
        if freq[d] > 10000:  # 滤除频率10000以上部分
            freqdata[d]=0
    return freqdata

def noiser(freq,d,freqdata): # 制造噪音
    while d:
        d-=1
        if freq[d] >= 11000: #在11000处添强烈噪音
            freqdata[d]=20000 #噪音强度要足够高效果才明显
    return freqdata

def wavwrite(path, nchannels, sampwidth, framerate, wave_data):
    # 打开WAV文档
    wavfile = we.open(path, "wb")
    # 配置声道数、量化位数和取样频率
    wavfile.setnchannels(nchannels)
    wavfile.setsampwidth(sampwidth)
    wavfile.setframerate(framerate)
    # 将wav_data转换为二进制数据写入文件
    wavfile.writeframes(wave_data.tostring())
    wavfile.close()


def main():
    path = 'eva.wav' #input("The Path is:")             #这是一个单声道的wav文件
    # 读取文件
    wavdata,wavtime,nchannels, sampwidth, framerate, nframes = wavread(path)

    freq,d,freqdata = getfreq(wavdata,nframes,framerate,0) # 获得频域信号
    # 滤波前波形绘制
    plt.subplot(421)                                    
    plt.plot(wavtime, wavdata[0],color = 'green')       #时域
    plt.subplot(422)                                    
    plt.plot(freq[:d-1],abs(freqdata[:d-1]),'r')        #频域

    freqdata_n = noiser(freq,d,freqdata)                #噪声加入！
    data_n = freq2time(nframes,freqdata_n)              #fft逆变换将噪声频域转时域
    # 噪声后波形绘制
    plt.subplot(423)                                    
    plt.plot(wavtime, data_n, color = 'green')          #噪声后时域
    plt.subplot(424)                                    
    plt.plot(freq[:d-1],abs(freqdata_n[:d-1]),'r')      #噪声后频域

    # 写入加入噪声的文件
    wavwrite('evan.wav', nchannels, sampwidth, framerate, data_n)

    freqdata_f = filter(freq,d,freqdata)                #滤波
    data_f = freq2time(nframes,freqdata_f)              #fft逆变换转回时域
    # 滤波后波形绘制
    plt.subplot(413)                                    
    plt.plot(wavtime, data_f, color = 'green')          #滤波后时域
    plt.subplot(414)                                    
    plt.plot(freq[:d-1],abs(freqdata_f[:d-1]),'r')      #滤波后频域

    # 写入滤除噪声后的文件
    wavwrite('evaf.wav', nchannels, sampwidth, framerate, data_f)

    plt.show()




if __name__ == "__main__":
    main()