import pickle
import numpy as np
import os # to change the working directory

#### for epoch based analysis
# Analysis="30s"

# objectsCh2 = []
# with (open("AphaseInde"+Analysis+".txt", "rb")) as openfile:
#     while True:
#         try:
#             objectsCh2.append(pickle.load(openfile))
#         except EOFError:
#             break
# AphaseInde30s=np.asarray(objectsCh2[0])
# for i in range (1, 19, 1):
#     AphaseInde30s=np.concatenate((AphaseInde30s, np.asarray(objectsCh2[i])), axis=0)
# objectsCh2 = []
# with (open("AphaseInde"+Analysis+"d.txt", "rb")) as openfile:
#     while True:
#         try:
#             objectsCh2.append(pickle.load(openfile))
#         except EOFError:
#             break
# AphaseInde30sd=np.asarray(objectsCh2[0])
# for i in range (1, 19, 1):
#     AphaseInde30sd=np.concatenate((AphaseInde30sd, np.asarray(objectsCh2[i])), axis=0)
    
    

#### for subject based analysis
Analysis="60s"
useSD=1 # 1 to use the SD subejct
disorder = "nfle" # specify the examined disorder, ins for insomnia, nfle for NFLE
os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory

if useSD == 1:
    objectsCh2 = []
    with (open(str(disorder)+"AphaseInde"+Analysis+"BLplotSD.txt", "rb")) as openfile:
        while True:
            try:
                objectsCh2.append(pickle.load(openfile))
            except EOFError:
                break
    AphaseIndex=np.asarray(objectsCh2)
    objectsCh2 = []
    with (open(str(disorder)+"AphaseInde"+Analysis+"dBLplotSD.txt", "rb")) as openfile:
        while True:
            try:
                objectsCh2.append(pickle.load(openfile))
            except EOFError:
                break
    AphaseIndexd=np.asarray(objectsCh2)
else:
    objectsCh2 = []
    with (open(str(disorder)+"AphaseInde"+Analysis+"BLplot.txt", "rb")) as openfile:
        while True:
            try:
                objectsCh2.append(pickle.load(openfile))
            except EOFError:
                break
    AphaseIndex=np.asarray(objectsCh2)
    objectsCh2 = []
    with (open(str(disorder)+"AphaseInde"+Analysis+"dBLplot.txt", "rb")) as openfile:
        while True:
            try:
                objectsCh2.append(pickle.load(openfile))
            except EOFError:
                break
    AphaseIndexd=np.asarray(objectsCh2)
    
    
    
    
# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt
# m1 = AphaseIndex
# m2 = AphaseIndexd
# f, ax = plt.subplots(1, figsize = (10,10))
# sm.graphics.mean_diff_plot(m2, m1, ax = ax)
# plt.show()

# import pyCompare
# pyCompare.blandAltman(AphaseIndex, AphaseIndexd,
#                 limitOfAgreement=1.96,
#                 confidenceInterval=95,
#                 confidenceIntervalMethod='approximate',
#                 detrend=None,
#                 percentage=False,)

import pingouin as pg
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
ax = pg.plot_blandaltman(AphaseIndexd, AphaseIndex)
plt.rcParams.update({'font.size': 20})
plt.ylabel('Difference')
plt.xlabel('Mean')
plt.ylim([-0.2, 0.2])
plt.xlim([0, 0.4])