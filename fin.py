from __future__ import division
import librosa as li
import numpy as np 
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import stats
from sklearn.mixture import GMM
from math import *
from scipy.fftpack import fft
def checkstartwith(a,b):
    lena=len(a)
    b=b[:lena]
    a=a[:len(a)-1]
    print(b,a)
    if(b==a):
        print("They are matched at the start")
    else:
        fl=True
        i=0
        for  i in range(len(a)-1):
            if(a[i]==b[i]):
                fl=False
                break
        if(fl):
            print("They are matched at the start")
        else:
            print("They are not matched at the start")
def checkfull(a,b):
    if(b==a):
        print("They are matched")
    else:
        fl=True
        i=0
        for  i in range(len(a)-1):
            if(a[i]==b[i]):
                fl=False
                break
        if(fl):
            print("They are matched")
        else:
            print("They are not matched")
def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)):
                    answer = match
                match = ""
    return answer
def checkmiddle(a,b):
    perct=0.00000
    if(len(a)<len(b)):
        perct=(len(longestSubstringFinder(a,b))/len(a))*100
    else:
        perct=(len(longestSubstringFinder(a,b))/len(b))*100
    print(str(perct)+"% of song is matched")
audio_time_series, sample_rate = li.load(r"cello.wav")
length_series = len(audio_time_series)
zero_crossings = []
energy = []
entropy_of_energy = []
mfcc = []
chroma_stft = []
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0)]
     z = li.zero_crossings(frame_self)
     arr = np.nonzero(z)
     zero_crossings.append(len(arr[0]))
     mt = []
     mf = li.feature.mfcc(frame_self)
     for k in range(0,len(mf)):
         mt.append(np.mean(mf[k]))
     mfcc.append(mt)
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ct = []
     cf = li.feature.chroma_stft(frame_self)
     for k in range(0,len(cf)):
          ct.append(np.mean(cf[k]))
     ca=sum(ct)/len(ct)
     chroma_stft.append(ca)
f_list_1 = []
f_list_1.append(zero_crossings)
f_list_1.append(energy)
f_list_1.append(chroma_stft)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
f_np_3 = np.array(mfcc)
master = np.concatenate([f_np_1,f_np_3], axis=1)
clf=KMeans(n_clusters=2).fit(master)
seq=clf.predict(master)
a="".join([str(x) for x in seq])
audio_time_series, sample_rate = li.load(r"strings4.wav")
audio_time_series=audio_time_series
length_series = len(audio_time_series)
zero_crossings = []
energy = []
entropy_of_energy = []
mfcc = []
chroma_stft = []
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0)]
     z = li.zero_crossings(frame_self)
     arr = np.nonzero(z)
     zero_crossings.append(len(arr[0]))
     mt = []
     mf = li.feature.mfcc(frame_self)
     for k in range(0,len(mf)):
         mt.append(np.mean(mf[k]))
     mfcc.append(mt)
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ct = []
     cf = li.feature.chroma_stft(frame_self)
     for k in range(0,len(cf)):
          ct.append(np.mean(cf[k]))
     ca=sum(ct)/len(ct)
     chroma_stft.append(ca)
f_list_1 = []
f_list_1.append(zero_crossings)
f_list_1.append(energy)
f_list_1.append(chroma_stft)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)
f_np_3 = np.array(mfcc)
master = np.concatenate([f_np_1,f_np_3], axis=1)
clf=KMeans(n_clusters=2).fit(master)
seq=clf.predict(master)
b="".join([str(x) for x in seq])
checkmiddle(b,a)
