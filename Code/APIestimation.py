import pickle # to store the results of the examination in files
import numpy as np
import matplotlib.pyplot as plt
import math  
import matplotlib.pylab as plb  
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import os # to change the working directory

numberSubjectsN = 14 # number of normal subjects to be considered (from 0 to 14)
numberSubjectsSD = 3 # number of subjects with Sleep-Disordered (SD) to be considered (from 0 to 3)
startEpochs = 0 # number of the subject to start the leave one out examination
useSDpatients = 1 # 1 to use the SD patiens and 0 to not use the SD patients
UseAPhaseEnsable = 1 # 1 to use the model with the esamble of A phase classifiers and 0 to use the model with only the estimation based on the center (overlappinf at left and right)
disorder = "nfle" # specify the examined disorder, ins for insomnia, nfle for NFLE
if useSDpatients == 1:
    Epochs = 19 # number of the subject to finish the leave one out examination
    BeginTest = 18 # location of the sorted array where the testing subject for the leave one out examination is identified
else:
    Epochs = 19-numberSubjectsSD-1 # number of the subject to finish the leave one out examination
    BeginTest = 18-numberSubjectsSD-1 # location of the sorted array where the testing subject for the leave one out examination is identified
Begin = 0 # location of the sorted array where the data for the first subject used to compose the training dataset for the leave one out examination is identified, the training dataset is composed of subejcts from Begin to BeginTest
startTime = 0 # time where to start the analysis (for example, to elminate the first 30 minutes use 60*30)
ff = 0 # number of the epoch to be examined
PolinomialOrder = 1 # order of the polinomial to fit the regression line
plotData = 0 # 1 to create and save the plots 
method = 0 # method to estima the error percentage: 0 -> (np.mean(abs(AphaseIndex-AphaseIndexd)))/np.mean(AphaseIndexd)*100, 1 -> (abs(np.mean(AphaseIndex)-np.mean(AphaseIndexd)))/np.mean(AphaseIndexd)*100
KernelSize = 5 # number of kernels used by the CNN
os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory

numberSubjects=Epochs-startEpochs
indices = np.arange(numberSubjects)

API_metrics=np.zeros((Epochs,5))

def CalculateNerRealTIme (A, N, AphaseIndex):
    ContingA=0
    ContingN=0
    ContingAInd=0
    ContingNInd=0
    indes = 0
    for k in range (startTime, len(A), 1):
        if A[k] > 0:
            ContingA+=1
            ContingAInd+=1
        if N[k] > 0:
            ContingN+=1 
            ContingNInd+=1
        if ContingN > 0:
            AphaseIndex[k]=ContingA/ContingN
        indes+=1
    return AphaseIndex
        
def CalculateAccumulation (time, A, N, AphaseIndexInd):
    ContingA=0
    ContingN=0
    ContingAInd=0
    ContingNInd=0
    indes = 0
    inc=0
    for k in range (startTime, len(A), 1):
        if A[k] > 0:
            ContingA+=1
            ContingAInd+=1
        if N[k] > 0:
            ContingN+=1 
            ContingNInd+=1
        if indes == time:
            indes = 0
            if ContingNInd > 0:
                AphaseIndexInd[inc]=ContingAInd/ContingNInd
            ContingAInd=0
            ContingNInd=0 
            inc+=1
        indes+=1
    return AphaseIndexInd
    
def zero_to_nan (values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def ReadDataForAnalysis (StringText):
    objects = []
    with (open(StringText, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    # print(objects)
    return objects [-1]


def PlotSubject (ee,ff,AphaseIndex,AphaseInde30s,AphaseInde60s,
                 AphaseInde30m,AphaseInde60m,AphaseIndexd,AphaseInde30sd,
                 AphaseInde60sd,AphaseInde30md,AphaseInde60md,OriginalOrCorrected):
    if OriginalOrCorrected == 0:
        addText="BeforeCorrection"
    else:
        addText="AfterCorrection"
    # near real-time plot
    plt.ioff()
    plt.figure()
    plt.plot(zero_to_nan(AphaseIndex))
    plt.plot(zero_to_nan(AphaseIndexd))
    plt.ylabel('API - near real-time')
    plt.xlabel('Time (s)')
    plt.legend(["Predicted", "Database"])
    plt.title("Subejct{}_Epoch{}".format(ee, ff)) 
    plt.ylim([0, 1])
    plt.grid()
    if UseAPhaseEnsable == 1:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"NearRealTime_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"NearRealTime_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    else:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"NearRealTime_Center_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"NearRealTime_Center_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    plt.close()
    # plt.show()
    # 30 s accumulated plot
    plt.ioff()
    plt.figure()
    plt.plot(AphaseInde30s)
    plt.plot(AphaseInde30sd)
    plt.ylabel('AphaseIndex')
    plt.xlabel('Time (30 s)')
    plt.legend(["Predicted", "Database"])
    plt.title("Subejct{}_Epoch{}".format(ee, ff)) 
    plt.ylim([0, 1])
    plt.grid()
    if UseAPhaseEnsable == 1:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated30s_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated30s_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    else:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated30s_Center_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated30s_Center_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    # plt.show()
    # 60 s accumulated plot
    plt.close()
    plt.ioff()
    plt.figure()
    plt.plot(AphaseInde60s)
    plt.plot(AphaseInde60sd)
    plt.ylabel('AphaseIndex')
    plt.xlabel('Time (60 s)')
    plt.legend(["Predicted", "Database"])
    plt.title("Subejct{}_Epoch{}".format(ee, ff)) 
    plt.ylim([0, 1])
    plt.grid()
    if UseAPhaseEnsable == 1:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated60s_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated60s_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    else:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated60s_Center_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated60s_Center_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    # plt.show()
    # 30 m accumulated plot
    plt.close()
    plt.ioff()
    plt.figure()
    plt.plot(AphaseInde30m)
    plt.plot(AphaseInde30md)
    plt.ylabel('AphaseIndex')
    plt.xlabel('Time (30 min)')
    plt.legend(["Predicted", "Database"])
    plt.title("Subejct{}_Epoch{}".format(ee, ff)) 
    plt.ylim([0, 1])
    plt.grid()
    if UseAPhaseEnsable == 1:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated30m_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated30m_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    else:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated30m_Center_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated30m_Center_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    # plt.show()
    # 60 m accumulated plot
    plt.close()
    plt.ioff()
    plt.figure()
    plt.plot(AphaseInde60m)
    plt.plot(AphaseInde60md)
    plt.ylabel('AphaseIndex')
    plt.xlabel('Time (60 min)')
    plt.legend(["Predicted", "Database"])
    plt.title("Subejct{}_Epoch{}".format(ee, ff)) 
    plt.ylim([0, 1])
    plt.grid()
    if UseAPhaseEnsable == 1:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated60m_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated60m_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    else:
        if useSDpatients == 1:
            plt.savefig(str(disorder)+"Accumulated60m_Center_Subejct_"+addText+"_{}_Epoch{}.png".format(ee, ff))
        else:
            plt.savefig(str(disorder)+"Accumulated60m_Center_Subejct_"+addText+"_{}_Epoch{}-notSD.png".format(ee, ff))
    # plt.show()
    plt.close()

def plotRegression (API_MetricsAverages,API_MetricsAveragesDatabase,PolinomialOrder):
    # plt.ioff()
    # plt.figure()
            # plt.scatter(API_MetricsAverages[:,0],API_MetricsAveragesDatabase[:,0])   
            # z = np.polyfit(API_MetricsAverages[:,0],API_MetricsAveragesDatabase[:,0], PolinomialOrder)
            # p = np.poly1d(z)
            # x2 = np.linspace(0, np.max(API_MetricsAverages), 100)
            # y2 = p(x2)
            # plt.plot(x2, y2 , 'm-')
            # plt.ylim([0, np.ceil(np.max(API_MetricsAveragesDatabase[:,0]))])
            # plt.ylabel('API near real-time database')
            # plt.xlabel('API near real-time predicted')
            # plt.grid()
    df = pd.DataFrame((API_MetricsAverages[:,0], API_MetricsAveragesDatabase[:,0]))
    df = df.T
    df.columns = ['API near real-time predicted','API near real-time database']
    sns.jointplot(x="API near real-time predicted", y="API near real-time database", data=df, order=PolinomialOrder, kind="reg")
    sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
    sns.set_style("whitegrid")
    # sns.jointplot(API_MetricsAverages[:,0],API_MetricsAveragesDatabase[:,0], order=PolinomialOrder, kind="reg")
    plt.show()
    # if useSDpatients == 1:
    #     plt.savefig(str(disorder)+"Regression_API_NearRealTime.png")
    # else:
    #     plt.savefig(str(disorder)+"Regression_API_NearRealTime-notSD.png")
    # plt.close()
    # plt.ioff()
    # plt.figure()
            # plt.scatter(API_MetricsAverages[:,1],API_MetricsAveragesDatabase[:,1])   
            # z = np.polyfit(API_MetricsAverages[:,1],API_MetricsAveragesDatabase[:,1], PolinomialOrder)
            # p = np.poly1d(z)
            # x2 = np.linspace(0, np.max(API_MetricsAverages), 100)
            # y2 = p(x2)
            # plt.plot(x2, y2 , 'm-')
            # plt.ylim([0, np.ceil(np.max(API_MetricsAveragesDatabase[:,1]))])
            # plt.ylabel('API 30 s database')
            # plt.xlabel('API 30 s predicted')
            # plt.grid()
    df = pd.DataFrame((API_MetricsAverages[:,1], API_MetricsAveragesDatabase[:,1]))
    df = df.T
    df.columns = ['API 30 s predicted','API 30 s database']
    sns.jointplot(x="API 30 s predicted", y="API 30 s database", data=df, order=PolinomialOrder, kind="reg")
    sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
    sns.set_style("whitegrid")
    plt.show()
    # if useSDpatients == 1:
    #     plt.savefig(str(disorder)+"Regression_API_30s.png")
    # else:
    #     plt.savefig(str(disorder)+"Regression_API_30s-notSD.png")
    # plt.close()
    # plt.ioff()
    # plt.figure()
            # plt.scatter(API_MetricsAverages[:,2],API_MetricsAveragesDatabase[:,2])   
            # z = np.polyfit(API_MetricsAverages[:,2],API_MetricsAveragesDatabase[:,2], PolinomialOrder)
            # p = np.poly1d(z)
            # x2 = np.linspace(0, np.max(API_MetricsAverages), 100)
            # y2 = p(x2)
            # plt.plot(x2, y2 , 'm-')
            # plt.ylim([0, np.ceil(np.max(API_MetricsAveragesDatabase[:,2]))])
            # plt.ylabel('API 60 s database')
            # plt.xlabel('API 60 s predicted')
            # plt.grid()
    df = pd.DataFrame((API_MetricsAverages[:,2], API_MetricsAveragesDatabase[:,2]))
    df = df.T
    df.columns = ['API 60 s predicted','API 60 s database']
    sns.jointplot(x="API 60 s predicted", y="API 60 s database", data=df, order=PolinomialOrder, kind="reg")
    sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
    sns.set_style("whitegrid")
    plt.show()
    # if useSDpatients == 1:
    #     plt.savefig(str(disorder)+"Regression_API_60s.png")
    # else:
    #     plt.savefig(str(disorder)+"Regression_API_60s-notSD.png")
    # plt.close()
    # plt.ioff()
    # plt.figure()
            # plt.scatter(API_MetricsAverages[:,3],API_MetricsAveragesDatabase[:,3])   
            # z = np.polyfit(API_MetricsAverages[:,3],API_MetricsAveragesDatabase[:,3], PolinomialOrder)
            # p = np.poly1d(z)
            # x2 = np.linspace(0, np.max(API_MetricsAverages), 100)
            # y2 = p(x2)
            # plt.plot(x2, y2 , 'm-')
            # plt.ylim([0, np.ceil(np.max(API_MetricsAveragesDatabase[:,3]))])
            # plt.ylabel('API 30 m database')
            # plt.xlabel('API 30 m predicted')
            # plt.grid()
    df = pd.DataFrame((API_MetricsAverages[:,3], API_MetricsAveragesDatabase[:,3]))
    df = df.T
    df.columns = ['API 30 min predicted','API 30 min database']
    sns.jointplot(x="API 30 min predicted", y="API 30 min database", data=df, order=PolinomialOrder, kind="reg")
    sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
    sns.set_style("whitegrid")
    plt.show()
    # if useSDpatients == 1:
    #     plt.savefig(str(disorder)+"Regression_API_30m.png")
    # else:
    #     plt.savefig(str(disorder)+Regression_API_30m-notSD.png")
    # plt.close()
    # plt.ioff()
    # plt.figure()
            # plt.scatter(API_MetricsAverages[:,4],API_MetricsAveragesDatabase[:,4])   
            # z = np.polyfit(API_MetricsAverages[:,4],API_MetricsAveragesDatabase[:,4], PolinomialOrder)
            # p = np.poly1d(z)
            # x2 = np.linspace(0, np.max(API_MetricsAverages), 100)
            # y2 = p(x2)
            # plt.plot(x2, y2 , 'm-')
            # plt.ylim([0, np.ceil(np.max(API_MetricsAveragesDatabase[:,4]))])
            # plt.ylabel('API 60 m database')
            # plt.xlabel('API 60 m predicted')
            # plt.grid()
    df = pd.DataFrame((API_MetricsAverages[:,4], API_MetricsAveragesDatabase[:,4]))
    df = df.T
    df.columns = ['API 60 min predicted','API 60 min database']
    sns.jointplot(x="API 60 min predicted", y="API 60 min database", data=df, order=PolinomialOrder, kind="reg")
    sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
    sns.set_style("whitegrid")
    plt.show()
    # if useSDpatients == 1:
    #     plt.savefig(str(disorder)+"Regression_API_60m.png")
    # else:
    #     plt.savefig(str(disorder)+"Regression_API_60m-notSD.png")
    # plt.close()


def polyfitFunction(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    # results = {}
    # # Polynomial Coefficients
    # results['polynomial'] = coeffs.tolist()
    # # r-squared
    # p = np.poly1d(coeffs)
    # # fit values, and mean
    # yhat = p(x)                         # or [p(z) for z in x]
    # ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    # ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    # sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    # results['determination'] = ssreg / sstot
    results = r2_score(y, p(x)) # coefficient of dermination: R^2
    return p, results

API_PencentageEror=np.zeros((Epochs,5))
API_MetricsAverages=np.zeros((Epochs,5))
API_MetricsAveragesDatabase=np.zeros((Epochs,5))
API_PencentageErorAfterCorrection=np.zeros((Epochs,5))
API_MetricsAveragesAfterCorrection=np.zeros((Epochs,5))


for ee in range (startEpochs, Epochs, 1): # number of the subject to be examined

    time30s = 30 # value in seconds for the acumulation
    time60s = 60 
    time30m = 60*30 
    time60m = 60*60 
    
    # From the model predictions
    
    if UseAPhaseEnsable == 1:
        StringTextA = str(disorder)+"EstimatedAPhase_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"EstimatedNREM_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    else:
        StringTextA = str(disorder)+"EstimatedAPhaseAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"EstimatedNREMAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 

    A = ReadDataForAnalysis (StringTextA)
    N = ReadDataForAnalysis (StringTextN)
    
    AphaseIndex = np.zeros((len(A)))
    AphaseIndex = CalculateNerRealTIme (A, N, AphaseIndex)
    AphaseInde30s = np.zeros(math.ceil((len(N)/(time30s)))-1)
    AphaseInde30s = CalculateAccumulation (time30s, A, N, AphaseInde30s)
    AphaseInde60s = np.zeros(math.ceil((len(N)/(time60s)))-1)
    AphaseInde60s = CalculateAccumulation (time60s, A, N, AphaseInde60s)
    AphaseInde30m = np.zeros(math.ceil((len(N)/(time30m)))-1)
    AphaseInde30m = CalculateAccumulation (time30m, A, N, AphaseInde30m)
    AphaseInde60m = np.zeros(math.ceil((len(N)/(time60m)))-1)
    AphaseInde60m = CalculateAccumulation (time60m, A, N, AphaseInde60m)

    # From database
    if UseAPhaseEnsable == 1:
        StringTextA = str(disorder)+"DatabaseAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"DatabaseNREM_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    else:
        StringTextA = str(disorder)+"DatabaseAPhaseAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"DatabaseNREMAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    Ad = ReadDataForAnalysis (StringTextA)
    Nd = ReadDataForAnalysis (StringTextN)
    
    AphaseIndexd = np.zeros((len(Ad)))
    AphaseIndexd = CalculateNerRealTIme (Ad, Nd, AphaseIndexd)
    AphaseInde30sd = np.zeros(math.ceil((len(Nd)/(time30s)))-1)
    AphaseInde30sd = CalculateAccumulation (time30s, Ad, Nd, AphaseInde30sd)
    AphaseInde60sd = np.zeros(math.ceil((len(Nd)/(time60s)))-1)
    AphaseInde60sd = CalculateAccumulation (time60s, Ad, Nd, AphaseInde60sd)
    AphaseInde30md = np.zeros(math.ceil((len(Nd)/(time30m)))-1)
    AphaseInde30md = CalculateAccumulation (time30m, Ad, Nd, AphaseInde30md)
    AphaseInde60md = np.zeros(math.ceil((len(Nd)/(time60m)))-1)
    AphaseInde60md = CalculateAccumulation (time60m, Ad, Nd, AphaseInde60md)
    
    AphaseIndexd[AphaseIndexd>1]=0
    AphaseInde30sd[AphaseInde30sd>1]=0
    AphaseInde60sd[AphaseInde60sd>1]=0
    AphaseInde30md[AphaseInde30md>1]=0
    AphaseInde60md[AphaseInde60md>1]=0

    if plotData == 1:
        PlotSubject (ee,ff,AphaseIndex,AphaseInde30s,AphaseInde60s,
                     AphaseInde30m,AphaseInde60m,AphaseIndexd,AphaseInde30sd,
                     AphaseInde60sd,AphaseInde30md,AphaseInde60md,0)
    
    if method == 1:
        API_PencentageEror[ee,0] = (np.mean(abs(AphaseIndex-AphaseIndexd)))/np.mean(AphaseIndexd)*100
        API_PencentageEror[ee,1] = (np.mean(abs(AphaseInde30s-AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
        API_PencentageEror[ee,2] = (np.mean(abs(AphaseInde60s-AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
        API_PencentageEror[ee,3] = (np.mean(abs(AphaseInde30m-AphaseInde30md)))/np.mean(AphaseInde30md)*100
        API_PencentageEror[ee,4] = (np.mean(abs(AphaseInde60m-AphaseInde60md)))/np.mean(AphaseInde60md)*100
    else:
        API_PencentageEror[ee,0] = (abs(np.mean(AphaseIndex)-np.mean(AphaseIndexd)))/np.mean(AphaseIndexd)*100
        API_PencentageEror[ee,1] = (abs(np.mean(AphaseInde30s)-np.mean(AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
        API_PencentageEror[ee,2] = (abs(np.mean(AphaseInde60s)-np.mean(AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
        API_PencentageEror[ee,3] = (abs(np.mean(AphaseInde30m)-np.mean(AphaseInde30md)))/np.mean(AphaseInde30md)*100
        API_PencentageEror[ee,4] = (abs(np.mean(AphaseInde60m)-np.mean(AphaseInde60md)))/np.mean(AphaseInde60md)*100

    API_MetricsAverages[ee,0] = np.mean(AphaseIndex)*100
    API_MetricsAverages[ee,1] = np.mean(AphaseInde30s)*100
    API_MetricsAverages[ee,2] = np.mean(AphaseInde60s)*100
    API_MetricsAverages[ee,3] = np.mean(AphaseInde30m)*100
    API_MetricsAverages[ee,4] = np.mean(AphaseInde60m)*100
    
    
    API_MetricsAveragesDatabase[ee,0] = np.mean(AphaseIndexd)*100
    API_MetricsAveragesDatabase[ee,1] = np.mean(AphaseInde30sd)*100
    API_MetricsAveragesDatabase[ee,2] = np.mean(AphaseInde60sd)*100
    API_MetricsAveragesDatabase[ee,3] = np.mean(AphaseInde30md)*100
    API_MetricsAveragesDatabase[ee,4] = np.mean(AphaseInde60md)*100



for ee in range (startEpochs, Epochs, 1): # number of the subject to be examined

    if PolinomialOrder > 0:
        if PolinomialOrder > 0:
            polinomialRegression=np.zeros((PolinomialOrder+1,5))
            polinomialRegressionR2=np.zeros((5))
        
        p0, R2_0 = polyfitFunction(API_MetricsAverages[indices != ee,0] /100,API_MetricsAveragesDatabase[indices != ee,0]/100, PolinomialOrder)
        p1, R2_1 = polyfitFunction(API_MetricsAverages[indices != ee,1]/100,API_MetricsAveragesDatabase[indices != ee,1]/100, PolinomialOrder)
        p2, R2_2 = polyfitFunction(API_MetricsAverages[indices != ee,2]/100,API_MetricsAveragesDatabase[indices != ee,2]/100, PolinomialOrder)
        p3, R2_3 = polyfitFunction(API_MetricsAverages[indices != ee,3]/100,API_MetricsAveragesDatabase[indices != ee,3]/100, PolinomialOrder)
        p4, R2_4 = polyfitFunction(API_MetricsAverages[indices != ee,4]/100,API_MetricsAveragesDatabase[indices != ee,4]/100, PolinomialOrder)
        # z = np.polyfit(API_MetricsAverages[:,0]/100,API_MetricsAveragesDatabase[:,0]/100, PolinomialOrder)
        # p0 = np.poly1d(z)
        # z = np.polyfit(API_MetricsAverages[:,1]/100,API_MetricsAveragesDatabase[:,1]/100, PolinomialOrder)
        # p1 = np.poly1d(z)
        # z = np.polyfit(API_MetricsAverages[:,2]/100,API_MetricsAveragesDatabase[:,2]/100, PolinomialOrder)
        # p2 = np.poly1d(z)
        # z = np.polyfit(API_MetricsAverages[:,3]/100,API_MetricsAveragesDatabase[:,3]/100, PolinomialOrder)
        # p3 = np.poly1d(z)
        # z = np.polyfit(API_MetricsAverages[:,4]/100,API_MetricsAveragesDatabase[:,4]/100, PolinomialOrder)
        # p4 = np.poly1d(z)
        for pp in range (0,PolinomialOrder+1,1):
            polinomialRegression[pp,0]=p0[pp]
            polinomialRegression[pp,1]=p1[pp]
            polinomialRegression[pp,2]=p2[pp]
            polinomialRegression[pp,3]=p3[pp]
            polinomialRegression[pp,4]=p4[pp]
        polinomialRegressionR2[0]=R2_0
        polinomialRegressionR2[1]=R2_1
        polinomialRegressionR2[2]=R2_2
        polinomialRegressionR2[3]=R2_3
        polinomialRegressionR2[4]=R2_4

    time30s = 30 # value in seconds for the acumulation
    time60s = 60 
    time30m = 60*30 
    time60m = 60*60 
    
    # From the model predictions
    
    if UseAPhaseEnsable == 1:
        StringTextA = str(disorder)+"EstimatedAPhase_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"EstimatedNREM_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    else:
        StringTextA = str(disorder)+"EstimatedAPhaseAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"EstimatedNREMAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 

    A = ReadDataForAnalysis (StringTextA)
    N = ReadDataForAnalysis (StringTextN)
    
    AphaseIndex = np.zeros((len(A)))
    AphaseIndex = CalculateNerRealTIme (A, N, AphaseIndex)
    AphaseInde30s = np.zeros(math.ceil((len(N)/(time30s)))-1)
    AphaseInde30s = CalculateAccumulation (time30s, A, N, AphaseInde30s)
    AphaseInde60s = np.zeros(math.ceil((len(N)/(time60s)))-1)
    AphaseInde60s = CalculateAccumulation (time60s, A, N, AphaseInde60s)
    AphaseInde30m = np.zeros(math.ceil((len(N)/(time30m)))-1)
    AphaseInde30m = CalculateAccumulation (time30m, A, N, AphaseInde30m)
    AphaseInde60m = np.zeros(math.ceil((len(N)/(time60m)))-1)
    AphaseInde60m = CalculateAccumulation (time60m, A, N, AphaseInde60m)

    # From database
    if UseAPhaseEnsable == 1:
        StringTextA = str(disorder)+"DatabaseAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"DatabaseNREM_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    else:
        StringTextA = str(disorder)+"DatabaseAPhaseAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
        StringTextN = str(disorder)+"DatabaseNREMAfterCorrection_Center_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    Ad = ReadDataForAnalysis (StringTextA)
    Nd = ReadDataForAnalysis (StringTextN)
    
    AphaseIndexd = np.zeros((len(Ad)))
    AphaseIndexd = CalculateNerRealTIme (Ad, Nd, AphaseIndexd)
    AphaseInde30sd = np.zeros(math.ceil((len(Nd)/(time30s)))-1)
    AphaseInde30sd = CalculateAccumulation (time30s, Ad, Nd, AphaseInde30sd)
    AphaseInde60sd = np.zeros(math.ceil((len(Nd)/(time60s)))-1)
    AphaseInde60sd = CalculateAccumulation (time60s, Ad, Nd, AphaseInde60sd)
    AphaseInde30md = np.zeros(math.ceil((len(Nd)/(time30m)))-1)
    AphaseInde30md = CalculateAccumulation (time30m, Ad, Nd, AphaseInde30md)
    AphaseInde60md = np.zeros(math.ceil((len(Nd)/(time60m)))-1)
    AphaseInde60md = CalculateAccumulation (time60m, Ad, Nd, AphaseInde60md)
    
    AphaseIndexd[AphaseIndexd>1]=0
    AphaseInde30sd[AphaseInde30sd>1]=0
    AphaseInde60sd[AphaseInde60sd>1]=0
    AphaseInde30md[AphaseInde30md>1]=0
    AphaseInde60md[AphaseInde60md>1]=0
  
    if method == 1:
        # API_PencentageErorAfterCorrection[ee,0] = (np.mean(abs(AphaseIndex*0.878-AphaseIndexd)))/np.mean(AphaseInde30sd)*100
        # API_PencentageErorAfterCorrection[ee,1] = (np.mean(abs(AphaseInde30s*0.919-AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
        # API_PencentageErorAfterCorrection[ee,2] = (np.mean(abs(AphaseInde60s*0.911-AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
        # API_PencentageErorAfterCorrection[ee,3] = (np.mean(abs(AphaseInde30m*0.91-AphaseInde30md)))/np.mean(AphaseInde30md)*100
        # API_PencentageErorAfterCorrection[ee,4] = (np.mean(abs(AphaseInde60m*0.905-AphaseInde60md)))/np.mean(AphaseInde60md)*100
        if PolinomialOrder <= 0:
            API_PencentageErorAfterCorrection[ee,0] = (np.mean(abs(AphaseIndex-AphaseIndexd)))/np.mean(AphaseIndexd)*100
            API_PencentageErorAfterCorrection[ee,1] = (np.mean(abs(AphaseInde30s-AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
            API_PencentageErorAfterCorrection[ee,2] = (np.mean(abs(AphaseInde60s-AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
            API_PencentageErorAfterCorrection[ee,3] = (np.mean(abs(AphaseInde30m-AphaseInde30md)))/np.mean(AphaseInde30md)*100
            API_PencentageErorAfterCorrection[ee,4] = (np.mean(abs(AphaseInde60m-AphaseInde60md)))/np.mean(AphaseInde60md)*100
        else:
            AphaseIndexCorrected = 0
            AphaseInde30sCorrected = 0
            AphaseInde60sCorrected = 0
            AphaseInde30mCorrected = 0
            AphaseInde60mCorrected = 0
            for PolyInterpCoef in range (0,PolinomialOrder+1,1):
                AphaseIndexCorrected = AphaseIndexCorrected + (np.mean(AphaseIndex) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,0]
                AphaseInde30sCorrected = AphaseInde30sCorrected + (np.mean(AphaseInde30s) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,1]
                AphaseInde60sCorrected = AphaseInde60sCorrected + (np.mean(AphaseInde60s) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,2]
                AphaseInde30mCorrected = AphaseInde30mCorrected + (np.mean(AphaseInde30m) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,3]
                AphaseInde60mCorrected = AphaseInde60mCorrected + (np.mean(AphaseInde60m) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,4]
            API_PencentageErorAfterCorrection[ee,0] = (np.mean(abs(AphaseIndexCorrected-AphaseIndexd)))/np.mean(AphaseIndexd)*100
            API_PencentageErorAfterCorrection[ee,1] = (np.mean(abs(AphaseInde30sCorrected-AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
            API_PencentageErorAfterCorrection[ee,2] = (np.mean(abs(AphaseInde60sCorrected-AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
            API_PencentageErorAfterCorrection[ee,3] = (np.mean(abs(AphaseInde30mCorrected-AphaseInde30md)))/np.mean(AphaseInde30md)*100
            API_PencentageErorAfterCorrection[ee,4] = (np.mean(abs(AphaseInde60mCorrected-AphaseInde60md)))/np.mean(AphaseInde60md)*100
            
            API_MetricsAveragesAfterCorrection[ee,0] = AphaseIndexCorrected*100
            API_MetricsAveragesAfterCorrection[ee,1] = AphaseInde30sCorrected*100
            API_MetricsAveragesAfterCorrection[ee,2] = AphaseInde60sCorrected*100
            API_MetricsAveragesAfterCorrection[ee,3] = AphaseInde30mCorrected*100
            API_MetricsAveragesAfterCorrection[ee,4] = AphaseInde60mCorrected*100
    else:
        if PolinomialOrder <= 0:
            API_PencentageErorAfterCorrection[ee,0] = (abs(np.mean(AphaseIndex)-np.mean(AphaseIndexd)))/np.mean(AphaseIndexd)*100
            API_PencentageErorAfterCorrection[ee,1] = (abs(np.mean(AphaseInde30s)-np.mean(AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
            API_PencentageErorAfterCorrection[ee,2] = (abs(np.mean(AphaseInde60s)-np.mean(AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
            API_PencentageErorAfterCorrection[ee,3] = (abs(np.mean(AphaseInde30m)-np.mean(AphaseInde30md)))/np.mean(AphaseInde30md)*100
            API_PencentageErorAfterCorrection[ee,4] = (abs(np.mean(AphaseInde60m)-np.mean(AphaseInde60md)))/np.mean(AphaseInde60md)*100
        else:
            AphaseIndexCorrected = 0
            AphaseInde30sCorrected = 0
            AphaseInde60sCorrected = 0
            AphaseInde30mCorrected = 0
            AphaseInde60mCorrected = 0
            for PolyInterpCoef in range (0,PolinomialOrder+1,1):
                AphaseIndexCorrected = AphaseIndexCorrected + (np.mean(AphaseIndex) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,0]
                AphaseInde30sCorrected = AphaseInde30sCorrected + (np.mean(AphaseInde30s) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,1]
                AphaseInde60sCorrected = AphaseInde60sCorrected + (np.mean(AphaseInde60s) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,2]
                AphaseInde30mCorrected = AphaseInde30mCorrected + (np.mean(AphaseInde30m) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,3]
                AphaseInde60mCorrected = AphaseInde60mCorrected + (np.mean(AphaseInde60m) ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,4]
                
            API_PencentageErorAfterCorrection[ee,0] = (abs(AphaseIndexCorrected-np.mean(AphaseIndexd)))/np.mean(AphaseIndexd)*100
            API_PencentageErorAfterCorrection[ee,1] = (abs(AphaseInde30sCorrected-np.mean(AphaseInde30sd)))/np.mean(AphaseInde30sd)*100
            API_PencentageErorAfterCorrection[ee,2] = (abs(AphaseInde60sCorrected-np.mean(AphaseInde60sd)))/np.mean(AphaseInde60sd)*100
            API_PencentageErorAfterCorrection[ee,3] = (abs(AphaseInde30mCorrected-np.mean(AphaseInde30md)))/np.mean(AphaseInde30md)*100
            API_PencentageErorAfterCorrection[ee,4] = (abs(AphaseInde60mCorrected-np.mean(AphaseInde60md)))/np.mean(AphaseInde60md)*100
    
            API_MetricsAveragesAfterCorrection[ee,0] = AphaseIndexCorrected*100
            API_MetricsAveragesAfterCorrection[ee,1] = AphaseInde30sCorrected*100
            API_MetricsAveragesAfterCorrection[ee,2] = AphaseInde60sCorrected*100
            API_MetricsAveragesAfterCorrection[ee,3] = AphaseInde30mCorrected*100
            API_MetricsAveragesAfterCorrection[ee,4] = AphaseInde60mCorrected*100
    
    API_metrics[ee,0]=AphaseIndexCorrected
    API_metrics[ee,1]=AphaseInde30sCorrected
    API_metrics[ee,2]=AphaseInde60sCorrected
    API_metrics[ee,3]=AphaseInde30mCorrected
    API_metrics[ee,4]=AphaseInde60mCorrected
    
    f = open(str(disorder)+"AphaseIndex.txt", 'ab')
    pickle.dump(AphaseIndex, f)
    f.close()
    f = open(str(disorder)+"AphaseIndexd.txt", 'ab')
    pickle.dump(AphaseIndexd, f)
    f.close()
    f = open(str(disorder)+"AphaseInde30s.txt", 'ab')
    pickle.dump(AphaseInde30s, f)
    f.close()
    f = open(str(disorder)+"AphaseInde30sd.txt", 'ab')
    pickle.dump(AphaseInde30sd, f)
    f.close()
    f = open(str(disorder)+"AphaseInde60s.txt", 'ab')
    pickle.dump(AphaseInde60s, f)
    f.close()
    f = open(str(disorder)+"AphaseInde60sd.txt", 'ab')
    pickle.dump(AphaseInde60sd, f)
    f.close()
    f = open(str(disorder)+"AphaseInde30m.txt", 'ab')
    pickle.dump(AphaseInde30m, f)
    f.close()
    f = open(str(disorder)+"AphaseInde30md.txt", 'ab')
    pickle.dump(AphaseInde30md, f)
    f.close()
    f = open(str(disorder)+"AphaseInde60m.txt", 'ab')
    pickle.dump(AphaseInde60m, f)
    f.close()
    f = open(str(disorder)+"AphaseInde60md.txt", 'ab')
    pickle.dump(AphaseInde60md, f)
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    # f = open("AphaseIndexBLplot.txt", 'ab')
    # pickle.dump(AphaseIndexCorrected, f)
    # f.close()
    # f = open("AphaseIndexdBLplot.txt", 'ab')
    # pickle.dump(np.mean(AphaseIndexd), f)
    # f.close()
    # f = open("AphaseInde30sBLplot.txt", 'ab')
    # pickle.dump(AphaseInde30sCorrected, f)
    # f.close()
    # f = open("AphaseInde30SDLplot.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde30sd), f)
    # f.close()
    # f = open("AphaseInde60sBLplot.txt", 'ab')
    # pickle.dump(AphaseInde60sCorrected, f)
    # f.close()
    # f = open("AphaseInde60SDLplot.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde60sd), f)
    # f.close()
    # f = open("AphaseInde30mBLplot.txt", 'ab')
    # pickle.dump(AphaseInde30mCorrected, f)
    # f.close()
    # f = open("AphaseInde30mdBLplot.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde30md), f)
    # f.close()
    # f = open("AphaseInde60mBLplot.txt", 'ab')
    # pickle.dump(AphaseInde60mCorrected, f)
    # f.close()
    # f = open("AphaseInde60mdBLplot.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde60md), f)
    # f.close()
    
    # f = open("AphaseIndexBLplotSD.txt", 'ab')
    # pickle.dump(AphaseIndexCorrected, f)
    # f.close()
    # f = open("AphaseIndexdBLplotSD.txt", 'ab')
    # pickle.dump(np.mean(AphaseIndexd), f)
    # f.close()
    # f = open("AphaseInde30sBLplotSD.txt", 'ab')
    # pickle.dump(AphaseInde30sCorrected, f)
    # f.close()
    # f = open("AphaseInde30SDLplotSD.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde30sd), f)
    # f.close()
    # f = open("AphaseInde60sBLplotSD.txt", 'ab')
    # pickle.dump(AphaseInde60sCorrected, f)
    # f.close()
    # f = open("AphaseInde60SDLplotSD.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde60sd), f)
    # f.close()
    # f = open("AphaseInde30mBLplotSD.txt", 'ab')
    # pickle.dump(AphaseInde30mCorrected, f)
    # f.close()
    # f = open("AphaseInde30mdBLplotSD.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde30md), f)
    # f.close()
    # f = open("AphaseInde60mBLplotSD.txt", 'ab')
    # pickle.dump(AphaseInde60mCorrected, f)
    # f.close()
    # f = open("AphaseInde60mdBLplotSD.txt", 'ab')
    # pickle.dump(np.mean(AphaseInde60md), f)
    # f.close()
    
if plotData == 1:
        # AphaseIndexCorrected = 0
        # AphaseInde30sCorrected = 0
        # AphaseInde60sCorrected = 0
        # AphaseInde30mCorrected = 0
        # AphaseInde60mCorrected = 0
        # for PolyInterpCoef in range (0,PolinomialOrder+1,1):
        #     AphaseIndexCorrected = AphaseIndexCorrected + (AphaseIndex ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,0]
        #     AphaseInde30sCorrected = AphaseInde30sCorrected + (AphaseInde30s ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,1]
        #     AphaseInde60sCorrected = AphaseInde60sCorrected + (AphaseInde60s ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,2]
        #     AphaseInde30mCorrected = AphaseInde30mCorrected + (AphaseInde30m ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,3]
        #     AphaseInde60mCorrected = AphaseInde60mCorrected + (AphaseInde60m ** PolyInterpCoef) * polinomialRegression [PolyInterpCoef,4]
        # PlotSubject (ee,ff,AphaseIndexCorrected,AphaseInde30sCorrected,AphaseInde60sCorrected,
        #              AphaseInde30mCorrected,AphaseInde60mCorrected,AphaseIndexd,AphaseInde30sd,
        #              AphaseInde60sd,AphaseInde30md,AphaseInde60md,1) 
    plotRegression (API_MetricsAverages,API_MetricsAveragesDatabase,PolinomialOrder)
        

