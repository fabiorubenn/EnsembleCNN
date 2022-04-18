import scipy.io as spio
import numpy as np
import pickle
from scipy import stats
import os # to change the working directory

PlotNormal=0 # 1 to plot for normal subjects
PlotSDB=0 # 1 to plot for SDB subjects
PlotErrorBars=1 # 1 to plot the errorbars
disorder = "nfle" # specify the examined disorder, ins for insomnia, nfle for NFLE
os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory

objectNRT = []
with (open(str(disorder)+"AphaseIndex.txt", "rb")) as openfile:
    while True:
        try:
            objectNRT.append(pickle.load(openfile))
        except EOFError:
            break
objectNRTd = []
with (open(str(disorder)+"AphaseIndexd.txt", "rb")) as openfile:
    while True:
        try:
            objectNRTd.append(pickle.load(openfile))
        except EOFError:
            break
object30s = []
with (open(str(disorder)+"AphaseInde30s.txt", "rb")) as openfile:
    while True:
        try:
            object30s.append(pickle.load(openfile))
        except EOFError:
            break
object30sd = []
with (open(str(disorder)+"AphaseInde30sd.txt", "rb")) as openfile:
    while True:
        try:
            object30sd.append(pickle.load(openfile))
        except EOFError:
            break
object60s = []
with (open(str(disorder)+"AphaseInde60s.txt", "rb")) as openfile:
    while True:
        try:
            object60s.append(pickle.load(openfile))
        except EOFError:
            break
object60sd = []
with (open(str(disorder)+"AphaseInde60sd.txt", "rb")) as openfile:
    while True:
        try:
            object60sd.append(pickle.load(openfile))
        except EOFError:
            break
object30m = []
with (open(str(disorder)+"AphaseInde30m.txt", "rb")) as openfile:
    while True:
        try:
            object30m.append(pickle.load(openfile))
        except EOFError:
            break
object30md = []
with (open(str(disorder)+"AphaseInde30md.txt", "rb")) as openfile:
    while True:
        try:
            object30md.append(pickle.load(openfile))
        except EOFError:
            break
object60m = []
with (open(str(disorder)+"AphaseInde60m.txt", "rb")) as openfile:
    while True:
        try:
            object60m.append(pickle.load(openfile))
        except EOFError:
            break
object60md = []
with (open(str(disorder)+"AphaseInde60md.txt", "rb")) as openfile:
    while True:
        try:
            object60md.append(pickle.load(openfile))
        except EOFError:
            break
        
import matplotlib.pyplot as plt

#### normal subjects
if PlotNormal > 0:
    NRT=np.zeros((14,len(max(objectNRT, key=len))))
    for k in range (14):
        for l in range(0,len(objectNRT[k]),1):
            NRT[k,l]=np.asarray(objectNRT[k])[l]
    plt.figure()
    plt.plot(objectNRT[0], 'k--', alpha=0.2)
    plt.plot(objectNRT[1], 'k--', alpha=0.2)
    plt.plot(objectNRT[2], 'k--', alpha=0.2)
    plt.plot(objectNRT[3], 'k--', alpha=0.2)
    plt.plot(objectNRT[4], 'k--', alpha=0.2)
    plt.plot(objectNRT[5], 'k--', alpha=0.2)
    plt.plot(objectNRT[6], 'k--', alpha=0.2)
    plt.plot(objectNRT[7], 'k--', alpha=0.2)
    plt.plot(objectNRT[8], 'k--', alpha=0.2)
    plt.plot(objectNRT[9], 'k--', alpha=0.2)
    plt.plot(objectNRT[10], 'k--', alpha=0.2)
    plt.plot(objectNRT[11], 'k--', alpha=0.2)
    plt.plot(objectNRT[12], 'k--', alpha=0.2)
    plt.plot(objectNRT[13], 'k--', alpha=0.2)
    plt.plot(objectNRT[14], 'k--', alpha=0.2)
    NRT[NRT == 0] = np.nan
    plt.plot(np.nanmean(NRT,axis=0), 'r-')
    plt.ylabel('API near-real time')
    plt.xlabel('Time (s)')
    plt.xlim((0, 25000))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    
    NRTd=np.zeros((14,len(max(objectNRTd, key=len))))
    for k in range (14):
        for l in range(0,len(objectNRTd[k]),1):
            NRTd[k,l]=np.asarray(objectNRTd[k])[l]
    plt.figure()
    plt.plot(objectNRTd[0], 'k--', alpha=0.2)
    plt.plot(objectNRTd[1], 'k--', alpha=0.2)
    plt.plot(objectNRTd[2], 'k--', alpha=0.2)
    plt.plot(objectNRTd[3], 'k--', alpha=0.2)
    plt.plot(objectNRTd[4], 'k--', alpha=0.2)
    plt.plot(objectNRTd[5], 'k--', alpha=0.2)
    plt.plot(objectNRTd[6], 'k--', alpha=0.2)
    plt.plot(objectNRTd[7], 'k--', alpha=0.2)
    plt.plot(objectNRTd[8], 'k--', alpha=0.2)
    plt.plot(objectNRTd[9], 'k--', alpha=0.2)
    plt.plot(objectNRTd[10], 'k--', alpha=0.2)
    plt.plot(objectNRTd[11], 'k--', alpha=0.2)
    plt.plot(objectNRTd[12], 'k--', alpha=0.2)
    plt.plot(objectNRTd[13], 'k--', alpha=0.2)
    plt.plot(objectNRTd[14], 'k--', alpha=0.2)
    NRTd[NRTd == 0] = np.nan
    plt.plot(np.nanmean(NRTd,axis=0), 'r-')
    plt.ylabel('API near-real time - database')
    plt.xlabel('Time (s)')
    plt.xlim((0, 25000))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    
    API30s=np.zeros((14,len(max(object30s, key=len))))
    for k in range (14):
        for l in range(0,len(object30s[k]),1):
            API30s[k,l]=np.asarray(object30s[k])[l]
    plt.figure()
    plt.plot(object30s[0], 'k--', alpha=0.2)
    plt.plot(object30s[1], 'k--', alpha=0.2)
    plt.plot(object30s[2], 'k--', alpha=0.2)
    plt.plot(object30s[3], 'k--', alpha=0.2)
    plt.plot(object30s[4], 'k--', alpha=0.2)
    plt.plot(object30s[5], 'k--', alpha=0.2)
    plt.plot(object30s[6], 'k--', alpha=0.2)
    plt.plot(object30s[7], 'k--', alpha=0.2)
    plt.plot(object30s[8], 'k--', alpha=0.2)
    plt.plot(object30s[9], 'k--', alpha=0.2)
    plt.plot(object30s[10], 'k--', alpha=0.2)
    plt.plot(object30s[11], 'k--', alpha=0.2)
    plt.plot(object30s[12], 'k--', alpha=0.2)
    plt.plot(object30s[13], 'k--', alpha=0.2)
    plt.plot(object30s[14], 'k--', alpha=0.2)
    API30s[API30s == 0] = np.nan
    plt.plot(np.nanmean(API30s,axis=0), 'r-')
    plt.ylabel('API 30 s')
    plt.xlabel('Time (30 s)')
    plt.xlim((0, 1000))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API30sd=np.zeros((14,len(max(object30sd, key=len))))
    for k in range (14):
        for l in range(0,len(object30sd[k]),1):
            API30sd[k,l]=np.asarray(object30sd[k])[l]
    plt.figure()
    plt.plot(object30sd[0], 'k--', alpha=0.2)
    plt.plot(object30sd[1], 'k--', alpha=0.2)
    plt.plot(object30sd[2], 'k--', alpha=0.2)
    plt.plot(object30sd[3], 'k--', alpha=0.2)
    plt.plot(object30sd[4], 'k--', alpha=0.2)
    plt.plot(object30sd[5], 'k--', alpha=0.2)
    plt.plot(object30sd[6], 'k--', alpha=0.2)
    plt.plot(object30sd[7], 'k--', alpha=0.2)
    plt.plot(object30sd[8], 'k--', alpha=0.2)
    plt.plot(object30sd[9], 'k--', alpha=0.2)
    plt.plot(object30sd[10], 'k--', alpha=0.2)
    plt.plot(object30sd[11], 'k--', alpha=0.2)
    plt.plot(object30sd[12], 'k--', alpha=0.2)
    plt.plot(object30sd[13], 'k--', alpha=0.2)
    plt.plot(object30sd[14], 'k--', alpha=0.2)
    API30sd[API30sd == 0] = np.nan
    plt.plot(np.nanmean(API30sd,axis=0), 'r-')
    plt.ylabel('API 30 s - database')
    plt.xlabel('Time (30 s)')
    plt.xlim((0, 1000))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API60s=np.zeros((14,len(max(object60s, key=len))))
    for k in range (14):
        for l in range(0,len(object60s[k]),1):
            API60s[k,l]=np.asarray(object60s[k])[l]
    plt.figure()
    plt.plot(object60s[0], 'k--', alpha=0.2)
    plt.plot(object60s[1], 'k--', alpha=0.2)
    plt.plot(object60s[2], 'k--', alpha=0.2)
    plt.plot(object60s[3], 'k--', alpha=0.2)
    plt.plot(object60s[4], 'k--', alpha=0.2)
    plt.plot(object60s[5], 'k--', alpha=0.2)
    plt.plot(object60s[6], 'k--', alpha=0.2)
    plt.plot(object60s[7], 'k--', alpha=0.2)
    plt.plot(object60s[8], 'k--', alpha=0.2)
    plt.plot(object60s[9], 'k--', alpha=0.2)
    plt.plot(object60s[10], 'k--', alpha=0.2)
    plt.plot(object60s[11], 'k--', alpha=0.2)
    plt.plot(object60s[12], 'k--', alpha=0.2)
    plt.plot(object60s[13], 'k--', alpha=0.2)
    plt.plot(object60s[14], 'k--', alpha=0.2)
    API60s[API60s == 0] = np.nan
    plt.plot(np.nanmean(API60s,axis=0), 'r-')
    plt.ylabel('API 60 s')
    plt.xlabel('Time (60 s)')
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API60sd=np.zeros((14,len(max(object60sd, key=len))))
    for k in range (14):
        for l in range(0,len(object60sd[k]),1):
            API60sd[k,l]=np.asarray(object60sd[k])[l]
    plt.figure()
    plt.plot(object60sd[0], 'k--', alpha=0.2)
    plt.plot(object60sd[1], 'k--', alpha=0.2)
    plt.plot(object60sd[2], 'k--', alpha=0.2)
    plt.plot(object60sd[3], 'k--', alpha=0.2)
    plt.plot(object60sd[4], 'k--', alpha=0.2)
    plt.plot(object60sd[5], 'k--', alpha=0.2)
    plt.plot(object60sd[6], 'k--', alpha=0.2)
    plt.plot(object60sd[7], 'k--', alpha=0.2)
    plt.plot(object60sd[8], 'k--', alpha=0.2)
    plt.plot(object60sd[9], 'k--', alpha=0.2)
    plt.plot(object60sd[10], 'k--', alpha=0.2)
    plt.plot(object60sd[11], 'k--', alpha=0.2)
    plt.plot(object60sd[12], 'k--', alpha=0.2)
    plt.plot(object60sd[13], 'k--', alpha=0.2)
    plt.plot(object60sd[14], 'k--', alpha=0.2)
    API60sd[API60sd == 0] = np.nan
    plt.plot(np.nanmean(API60sd,axis=0), 'r-')
    plt.ylabel('API 60 s - database')
    plt.xlabel('Time (60 s)')
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API30m=np.zeros((14,len(max(object30m, key=len))))
    for k in range (14):
        for l in range(0,len(object30m[k]),1):
            API30m[k,l]=np.asarray(object30m[k])[l]
    plt.figure()
    plt.plot(object30m[0], 'k--', alpha=0.2)
    plt.plot(object30m[1], 'k--', alpha=0.2)
    plt.plot(object30m[2], 'k--', alpha=0.2)
    plt.plot(object30m[3], 'k--', alpha=0.2)
    plt.plot(object30m[4], 'k--', alpha=0.2)
    plt.plot(object30m[5], 'k--', alpha=0.2)
    plt.plot(object30m[6], 'k--', alpha=0.2)
    plt.plot(object30m[7], 'k--', alpha=0.2)
    plt.plot(object30m[8], 'k--', alpha=0.2)
    plt.plot(object30m[9], 'k--', alpha=0.2)
    plt.plot(object30m[10], 'k--', alpha=0.2)
    plt.plot(object30m[11], 'k--', alpha=0.2)
    plt.plot(object30m[12], 'k--', alpha=0.2)
    plt.plot(object30m[13], 'k--', alpha=0.2)
    plt.plot(object30m[14], 'k--', alpha=0.2)
    API30m[API30m == 0] = np.nan
    plt.plot(np.nanmean(API30m,axis=0), 'r-')
    plt.ylabel('API 30 m')
    plt.xlabel('Time (30 m)')
    plt.xlim((0, 14))
    plt.ylim((0, 0.7))
    plt.show()
    plt.grid()
    
    API30md=np.zeros((14,len(max(object30md, key=len))))
    for k in range (14):
        for l in range(0,len(object30md[k]),1):
            API30md[k,l]=np.asarray(object30md[k])[l]
    plt.figure()
    plt.plot(object30md[0], 'k--', alpha=0.2)
    plt.plot(object30md[1], 'k--', alpha=0.2)
    plt.plot(object30md[2], 'k--', alpha=0.2)
    plt.plot(object30md[3], 'k--', alpha=0.2)
    plt.plot(object30md[4], 'k--', alpha=0.2)
    plt.plot(object30md[5], 'k--', alpha=0.2)
    plt.plot(object30md[6], 'k--', alpha=0.2)
    plt.plot(object30md[7], 'k--', alpha=0.2)
    plt.plot(object30md[8], 'k--', alpha=0.2)
    plt.plot(object30md[9], 'k--', alpha=0.2)
    plt.plot(object30md[10], 'k--', alpha=0.2)
    plt.plot(object30md[11], 'k--', alpha=0.2)
    plt.plot(object30md[12], 'k--', alpha=0.2)
    plt.plot(object30md[13], 'k--', alpha=0.2)
    plt.plot(object30md[14], 'k--', alpha=0.2)
    API30md[API30md == 0] = np.nan
    plt.plot(np.nanmean(API30md,axis=0), 'r-')
    plt.ylabel('API 30 m - database')
    plt.xlabel('Time (30 m)')
    plt.xlim((0, 14))
    plt.ylim((0, 0.5))
    plt.show()
    plt.grid()
    
    API60m=np.zeros((14,len(max(object60m, key=len))))
    for k in range (14):
        for l in range(0,len(object60m[k]),1):
            API60m[k,l]=np.asarray(object60m[k])[l]
    plt.figure()
    plt.plot(object60m[0], 'k--', alpha=0.2)
    plt.plot(object60m[1], 'k--', alpha=0.2)
    plt.plot(object60m[2], 'k--', alpha=0.2)
    plt.plot(object60m[3], 'k--', alpha=0.2)
    plt.plot(object60m[4], 'k--', alpha=0.2)
    plt.plot(object60m[5], 'k--', alpha=0.2)
    plt.plot(object60m[6], 'k--', alpha=0.2)
    plt.plot(object60m[7], 'k--', alpha=0.2)
    plt.plot(object60m[8], 'k--', alpha=0.2)
    plt.plot(object60m[9], 'k--', alpha=0.2)
    plt.plot(object60m[10], 'k--', alpha=0.2)
    plt.plot(object60m[11], 'k--', alpha=0.2)
    plt.plot(object60m[12], 'k--', alpha=0.2)
    plt.plot(object60m[13], 'k--', alpha=0.2)
    plt.plot(object60m[14], 'k--', alpha=0.2)
    API60m[API60m == 0] = np.nan
    plt.plot(np.nanmean(API60m,axis=0), 'r-')
    plt.ylabel('API 60 m')
    plt.xlabel('Time (60 m)')
    plt.xlim((0, 7))
    plt.ylim((0, 0.7))
    plt.show()
    plt.grid()
    
    API60md=np.zeros((14,len(max(object60md, key=len))))
    for k in range (14):
        for l in range(0,len(object60md[k]),1):
            API60md[k,l]=np.asarray(object60md[k])[l]
    plt.figure()
    plt.plot(object60md[0], 'k--', alpha=0.2)
    plt.plot(object60md[1], 'k--', alpha=0.2)
    plt.plot(object60md[2], 'k--', alpha=0.2)
    plt.plot(object60md[3], 'k--', alpha=0.2)
    plt.plot(object60md[4], 'k--', alpha=0.2)
    plt.plot(object60md[5], 'k--', alpha=0.2)
    plt.plot(object60md[6], 'k--', alpha=0.2)
    plt.plot(object60md[7], 'k--', alpha=0.2)
    plt.plot(object60md[8], 'k--', alpha=0.2)
    plt.plot(object60md[9], 'k--', alpha=0.2)
    plt.plot(object60md[10], 'k--', alpha=0.2)
    plt.plot(object60md[11], 'k--', alpha=0.2)
    plt.plot(object60md[12], 'k--', alpha=0.2)
    plt.plot(object60md[13], 'k--', alpha=0.2)
    plt.plot(object60md[14], 'k--', alpha=0.2)
    API60md[API60md == 0] = np.nan
    plt.plot(np.nanmean(API60md,axis=0), 'r-')
    plt.ylabel('API 60 m - database')
    plt.xlabel('Time (60 m)')
    plt.xlim((0, 7))
    plt.ylim((0, 0.4))
    plt.show()
    plt.grid()


#### SDB subjects

if PlotSDB > 0:
    
    NRT=np.zeros((4,len(max(objectNRT, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(objectNRT[k]),1):
            NRT[k-15,l]=np.asarray(objectNRT[k])[l]
    plt.figure()
    plt.plot(objectNRT[15], 'k--', alpha=0.2)
    plt.plot(objectNRT[16], 'k--', alpha=0.2)
    plt.plot(objectNRT[17], 'k--', alpha=0.2)
    plt.plot(objectNRT[18], 'k--', alpha=0.2)
    NRT[NRT == 0] = np.nan
    plt.plot(np.nanmean(NRT,axis=0), 'r-')
    plt.ylabel('API near-real time')
    plt.xlabel('Time (s)')
    plt.xlim((0, 21600))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    
    NRTd=np.zeros((4,len(max(objectNRTd, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(objectNRTd[k]),1):
            NRTd[k-15,l]=np.asarray(objectNRTd[k])[l]
    plt.figure()
    plt.plot(objectNRTd[15], 'k--', alpha=0.2)
    plt.plot(objectNRTd[16], 'k--', alpha=0.2)
    plt.plot(objectNRTd[17], 'k--', alpha=0.2)
    plt.plot(objectNRTd[18], 'k--', alpha=0.2)
    NRTd[NRTd == 0] = np.nan
    plt.plot(np.nanmean(NRTd,axis=0), 'r-')
    plt.ylabel('API near-real time - database')
    plt.xlabel('Time (s)')
    plt.xlim((0, 21600))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    
    API30s=np.zeros((4,len(max(object30s, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object30s[k]),1):
            API30s[k-15,l]=np.asarray(object30s[k])[l]
    plt.figure()
    plt.plot(object30s[15], 'k--', alpha=0.2)
    plt.plot(object30s[16], 'k--', alpha=0.2)
    plt.plot(object30s[17], 'k--', alpha=0.2)
    plt.plot(object30s[18], 'k--', alpha=0.2)
    API30s[API30s == 0] = np.nan
    plt.plot(np.nanmean(API30s,axis=0), 'r-')
    plt.ylabel('API 30 s')
    plt.xlabel('Time (30 s)')
    plt.xlim((0, 720))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API30sd=np.zeros((4,len(max(object30sd, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object30sd[k]),1):
            API30sd[k-15,l]=np.asarray(object30sd[k])[l]
    plt.figure()
    plt.plot(object30sd[15], 'k--', alpha=0.2)
    plt.plot(object30sd[16], 'k--', alpha=0.2)
    plt.plot(object30sd[17], 'k--', alpha=0.2)
    plt.plot(object30sd[18], 'k--', alpha=0.2)
    API30sd[API30sd == 0] = np.nan
    plt.plot(np.nanmean(API30sd,axis=0), 'r-')
    plt.ylabel('API 30 s - database')
    plt.xlabel('Time (30 s)')
    plt.xlim((0, 720))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API60s=np.zeros((4,len(max(object60s, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60s[k]),1):
            API60s[k-15,l]=np.asarray(object60s[k])[l]
    plt.figure()
    plt.plot(object60s[15], 'k--', alpha=0.2)
    plt.plot(object60s[16], 'k--', alpha=0.2)
    plt.plot(object60s[17], 'k--', alpha=0.2)
    plt.plot(object60s[18], 'k--', alpha=0.2)
    API60s[API60s == 0] = np.nan
    plt.plot(np.nanmean(API60s,axis=0), 'r-')
    plt.ylabel('API 60 s')
    plt.xlabel('Time (60 s)')
    plt.xlim((0, 360))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API60sd=np.zeros((4,len(max(object60sd, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60sd[k]),1):
            API60sd[k-15,l]=np.asarray(object60sd[k])[l]
    plt.figure()
    plt.plot(object60sd[15], 'k--', alpha=0.2)
    plt.plot(object60sd[16], 'k--', alpha=0.2)
    plt.plot(object60sd[17], 'k--', alpha=0.2)
    plt.plot(object60sd[18], 'k--', alpha=0.2)
    API60sd[API60sd == 0] = np.nan
    plt.plot(np.nanmean(API60sd,axis=0), 'r-')
    plt.ylabel('API 60 s - database')
    plt.xlabel('Time (60 s)')
    plt.xlim((0, 360))
    plt.ylim((0, 1))
    plt.show()
    plt.grid()
    
    API30m=np.zeros((4,len(max(object30m, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object30m[k]),1):
            API30m[k-15,l]=np.asarray(object30m[k])[l]
    plt.figure()
    plt.plot(object30m[15], 'k--', alpha=0.2)
    plt.plot(object30m[16], 'k--', alpha=0.2)
    plt.plot(object30m[17], 'k--', alpha=0.2)
    plt.plot(object30m[18], 'k--', alpha=0.2)
    API30m[API30m == 0] = np.nan
    plt.plot(np.nanmean(API30m,axis=0), 'r-')
    plt.ylabel('API 30 m')
    plt.xlabel('Time (30 m)')
    plt.xlim((0, 12))
    plt.ylim((0, 0.8))
    plt.show()
    plt.grid()
    
    API30md=np.zeros((4,len(max(object30md, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object30md[k]),1):
            API30md[k-15,l]=np.asarray(object30md[k])[l]
    plt.figure()
    plt.plot(object30md[15], 'k--', alpha=0.2)
    plt.plot(object30md[16], 'k--', alpha=0.2)
    plt.plot(object30md[17], 'k--', alpha=0.2)
    plt.plot(object30md[18], 'k--', alpha=0.2)
    API30md[API30md == 0] = np.nan
    plt.plot(np.nanmean(API30md,axis=0), 'r-')
    plt.ylabel('API 30 m - database')
    plt.xlabel('Time (30 m)')
    plt.xlim((0, 12))
    plt.ylim((0, 0.8))
    plt.show()
    plt.grid()
    
    API60m=np.zeros((4,len(max(object60m, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60m[k]),1):
            API60m[k-15,l]=np.asarray(object60m[k])[l]
    plt.figure()
    plt.plot(object60m[15], 'k--', alpha=0.2)
    plt.plot(object60m[16], 'k--', alpha=0.2)
    plt.plot(object60m[17], 'k--', alpha=0.2)
    plt.plot(object60m[18], 'k--', alpha=0.2)
    API60m[API60m == 0] = np.nan
    plt.plot(np.nanmean(API60m,axis=0), 'r-')
    plt.ylabel('API 60 m')
    plt.xlabel('Time (60 m)')
    plt.xlim((0, 6))
    plt.ylim((0, 0.8))
    plt.show()
    plt.grid()
    
    API60md=np.zeros((4,len(max(object60md, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60md[k]),1):
            API60md[k-15,l]=np.asarray(object60md[k])[l]
    plt.figure()
    plt.plot(object60md[15], 'k--', alpha=0.2)
    plt.plot(object60md[16], 'k--', alpha=0.2)
    plt.plot(object60md[17], 'k--', alpha=0.2)
    plt.plot(object60md[18], 'k--', alpha=0.2)
    API60md[API60md == 0] = np.nan
    plt.plot(np.nanmean(API60md,axis=0), 'r-')
    plt.ylabel('API 60 m - database')
    plt.xlabel('Time (60 m)')
    plt.xlim((0, 6))
    plt.ylim((0, 0.7))
    plt.show()
    plt.grid()
    
if PlotErrorBars == 1:
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 45})
    API60m=np.zeros((4,len(max(object60m, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60m[k]),1):
            API60m[k-15,l]=np.asarray(object60m[k])[l]
    A=stats.sem(API60m, axis=0,nan_policy='omit')
    x = np.linspace(1, len(A), len(A), endpoint=True)
    API60m[API60m == 0] = np.nan
    plt.errorbar(x=x,y=np.nanmean(API60m,axis=0), yerr=A, ecolor='c', ls='none', elinewidth=10,alpha=0.2)
    API60md=np.zeros((4,len(max(object60md, key=len))))
    for k in range (15,19,1):
        for l in range(0,len(object60md[k]),1):
            API60md[k-15,l]=np.asarray(object60md[k])[l]
    API60md[API60md == 0] = np.nan
    B=stats.sem(API60md, axis=0,nan_policy='omit')
    plt.errorbar(x=x,y=np.nanmean(API60md,axis=0), yerr=B, ecolor='k', ls='none', elinewidth=10,alpha=0.2)
    plt.plot(x,np.nanmean(API60m,axis=0), 'c-', label = 'Predicted')
    plt.plot(x,np.nanmean(API60md,axis=0), 'k--', label = 'Database')
    

    
    plt.ylabel('API 60 min')
    plt.xlabel('Time (hours)')
    plt.xlim((1, 7))
    plt.ylim((0, 0.7))
    plt.legend()
    plt.show()
    plt.grid()
    
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 45})
    API60m=np.zeros((14,len(max(object60m, key=len))))
    for k in range (14):
        for l in range(0,len(object60m[k]),1):
            API60m[k,l]=np.asarray(object60m[k])[l]
    A=stats.sem(API60m, axis=0,nan_policy='omit')
    x = np.linspace(1, len(A), len(A), endpoint=True)
    API60m[API60m == 0] = np.nan
    plt.errorbar(x=x,y=np.nanmean(API60m,axis=0), yerr=A, ecolor='c', ls='none', elinewidth=10,alpha=0.2)
    API60md=np.zeros((14,len(max(object60md, key=len))))
    for k in range (14):
        for l in range(0,len(object60md[k]),1):
            API60md[k,l]=np.asarray(object60md[k])[l]
    API60md[API60md == 0] = np.nan
    B=stats.sem(API60md, axis=0,nan_policy='omit')
    plt.errorbar(x=x,y=np.nanmean(API60md,axis=0), yerr=B, ecolor='k', ls='none', elinewidth=10,alpha=0.2)
    plt.plot(x,np.nanmean(API60m,axis=0), 'c-', label = 'Predicted')
    plt.plot(x,np.nanmean(API60md,axis=0), 'k--', label = 'Database')
    
    plt.ylabel('API 60 min')
    plt.xlabel('Time (hours)')
    plt.xlim((1, 7))
    plt.ylim((0, 0.7))
    plt.legend()
    plt.show()
    plt.grid()
    
    