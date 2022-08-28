import os # to change the working directory
os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report # to perform the examination of the results
import numpy as np

from pycm import *

ee=0 #subject
thresholdValueEnsemble=1.5


"""
# Control variables
"""
numberSubjectsN = 14 # number of normal subjects to be considered (from 0 to numberSubjectsN)
numberSubjectsSD = 3 # number of subjects with sleep Disorder to be considered (from 0 to numberSubjectsSD)
startEpochs = 0 # 4 number of the subject to start the leave one out examination
useSDpatients = 5 # 0 to use only healthy, 1 to use healthy and SDB, 2 to use healthy and NFLE, 3 to use PLM, X to use ins
Begin = 0 # location of the sorted array where the data for the first subject used to compose the training dataset for the leave one out examination is identified, the training dataset is composed of subejcts from Begin to BeginTest
EpochsWork = 1 # number of examined iterations for each subejct
OverLap = [8, 12, 8] # number of overlapping seconds to be considered in the overlapping windows of A phase analysis (need to be 0 or an odd number), either [0] if no overlapping or [amount of overlapping for right, amount of overlapping for center, amount of overlapping for left] to use overlapping
OverLapH = [16, 14, 16] # number of overlapping seconds to be considered in the overlapping windows of NREM analysis (need to be 0 or an odd number), either [0] if no overlapping or [amount of overlapping for right, amount of overlapping for center, amount of overlapping for left] to use overlapping
thresholdEarlyStoping = 0.005 # minimum increasse in the Area Under the receiving operating Curve (AUC) considered by the early stopping procedure
patienteceValue = 10 # value used by the early stopping procedure to specify the number of training cycles without a minimum increasse of thresholdEarlyStoping befor stoping the examination
KernelSize = 5 # size of the kernel used by the convolution layer
APhaseSubtype = 3 # 0 for A and not-A; 1 for A1 and not-A1; 2 for A2 and not-A2; 3 for A3 and not-A3

"""
# variables to store the results of the examination
"""
if useSDpatients > 0:
    Epochs = 19 # number of the subject to finish the leave one out examination
    BeginTest = 18 # location of the sorted array where the testing subject for the leave one out examination is identified
    if useSDpatients == 1: 
        disorder = "sdb" # healthy and SDB
    elif useSDpatients == 2:
        disorder = "nfle" # healthy and NFLE
    elif useSDpatients == 3:
        disorder = "plm" # healthy and PLM
    else:
        disorder = "ins" # healthy and ins
else:
    disorder = "n"
    Epochs = 19-numberSubjectsSD-1 # number of the subject to finish the leave one out examination
    BeginTest = 18-numberSubjectsSD-1 # location of the sorted array where the testing subject for the leave one out examination is identified
    

AccAtInterAC = np.zeros (EpochsWork) # variable holding the accuracy for the A phase estimation after the classifier ensemble
SenAtInterAC = np.zeros (EpochsWork) # variable holding the sensitivity for the A phase estimation after the classifier ensemble
SpeAtInterAC = np.zeros (EpochsWork) # variable holding the specificity for the A phase estimation after the classifier ensemble
TPInterAC = np.zeros (EpochsWork) # variable holding the true positives for the A phase estimation after the classifier ensemble
TNInterAC = np.zeros (EpochsWork) # variable holding the true negatives for the A phase estimation after the classifier ensemble
FPInterAC = np.zeros (EpochsWork) # variable holding the false positives for the A phase estimation after the classifier ensemble
FNInterAC = np.zeros (EpochsWork) # variable holding the false negatives for the A phase estimation after the classifier ensemble
PPVInterAC = np.zeros (EpochsWork) # variable holding the positive predictive value for the A phase estimation after the classifier ensemble
NPVInterAC = np.zeros (EpochsWork) # variable holding the negative predictive value for the A phase estimation after the classifier ensemble
AtotalInter = np.zeros (EpochsWork) # variable holding the predicted duration of all A phases epochs

AccAtEndAC = np.zeros (Epochs) # variable holding the accuracy for the A phase estimation of each subejcts after the classifier ensemble
SenAtEndAC = np.zeros (Epochs) # variable holding the sensitivity for the A phase estimation of each subejcts after the classifier ensemble 
SpeAtEndAC = np.zeros (Epochs) # variable holding the specifiity for the A phase estimation of each subejcts after the classifier ensemble
AUCAtEndAC = np.zeros (Epochs) # variable holding the AUC for the A phase estimation of each subejcts after the classifier ensemble
TPEndAC = np.zeros (Epochs) # variable holding the true positives for the A phase estimation of each subejcts after the classifier ensemble
TNEndAC = np.zeros (Epochs) # variable holding the true negatives for the A phase estimation of each subejcts after the classifier ensemble
FPEndAC = np.zeros (Epochs) # variable holding the false positives for the A phase estimation of each subejcts after the classifier ensemble
FNEndAC = np.zeros (Epochs) # variable holding the false negatives for the A phase estimation of each subejcts after the classifier ensemble
PPVEndAC = np.zeros (Epochs) # variable holding the positive predictive value for the A phase estimation of each subejcts after the classifier ensemble
NPVEndAC = np.zeros (Epochs) # variable holding the negative predictive value for the A phase estimation of each subejcts after the classifier ensemble
AtotalEnd = np.zeros(Epochs) # variable holding the predicted duration of all A phases epochs of each subejcts after the classifier en



ff=0
KernelSize = 5

for ee in range (startEpochs, Epochs, 1): # examine from subejct identified by startEpochsup to the subejct identified by Epochs


    os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory

    objectNRT = []
    if useSDpatients == 0:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
    else:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
            
    PredictionYA3=objectNRT[0]
    
    
    objectNRT = []
    if useSDpatients == 0:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseAPhase_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
    else:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
            
    YTestOneLine=objectNRT[0]
    
 
    
    # objectNRT = []
    # if useSDpatients == 0:
    #     with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseNREM_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
    #         while True:
    #             try:
    #                 objectNRT.append(pickle.load(openfile))
    #             except EOFError:
    #                 break
    # else:
    #     with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseNREM_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
    #         while True:
    #             try:
    #                 objectNRT.append(pickle.load(openfile))
    #             except EOFError:
    #                 break
            
    # YTestOneLineNREM=objectNRT[0]
    
    
    
    cappredictiony_predhA = np.zeros (len (PredictionYA3 [0])) # arry that wil contain the majority vonting output
    for combinationA in range (0, len (cappredictiony_predhA), 1): # check line by line the output of the classifiers
        # cappredictiony_predhATemp = np.sum (PredictionYA3 [:, combinationA] * AUCAtInterA) # variable to hold the sum of all predictions of each epoch  for the majority voting
        cappredictiony_predhATemp = np.sum (PredictionYA3 [:, combinationA])
        if cappredictiony_predhATemp >= thresholdValueEnsemble: # majority voting, if two of the three classifiers predicted 1 then it is 1 otherwise leave the 0
            cappredictiony_predhA [combinationA] = 1
    
    
    
    
    
    objectNRT = []
    if useSDpatients == 0:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_CombinationStrategy_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            
        #os.chdir("D:\MatlabCode\PytonTests") # change the working directory
        #with (open("EstimatedNREM_CombinationStrategy_subject{}_Epoch{}-notSDB.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
        cappredictiony_predhN=objectNRT[0]
    else:
        with (open(str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_CombinationStrategy_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            
        #os.chdir("D:\MatlabCode\PytonTests") # change the working directory
        #with (open("EstimatedNREM_CombinationStrategy_subject{}_Epoch{}-notSDB.txt".format(ee, ff, KernelSize), "rb")) as openfile:
            
            while True:
                try:
                    objectNRT.append(pickle.load(openfile))
                except EOFError:
                    break
        cappredictiony_predhN=objectNRT[0]
    

    
    # for k in range (len (cappredictiony_predhN) - 1):
    #     if k > 0:
    #         if cappredictiony_predhN [k - 1] == 0 and cappredictiony_predhN [k] == 1 and cappredictiony_predhN [k + 1] == 0:
    #             cappredictiony_predhN [k] = 0
                
    # for k in range (len (cappredictiony_predhN)-1):
    #     if k > 0:
    #         if cappredictiony_predhN [k - 1] == 1 and cappredictiony_predhN [k] == 0 and cappredictiony_predhN [k + 1] == 1:
    #             cappredictiony_predhN [k] = 1  
    
    
    # for k in range (len(cappredictiony_predhN)): # decrese the missclassification by converting the classified A phases into not-A when the NREM classifiers indicates a period of REM/wake (A phase can only occur during NREM sleep)
    #     if cappredictiony_predhN[k]==0:
    #         cappredictiony_predhA[k]=0
    for k in range (len (cappredictiony_predhA) - 1):
        if k > 0:
            if cappredictiony_predhA [k - 1] == 0 and cappredictiony_predhA [k] == 1 and cappredictiony_predhA [k + 1] == 0:
                cappredictiony_predhA [k] = 0
                
    for k in range (len (cappredictiony_predhA)-1):
        if k > 0:
            if cappredictiony_predhA [k - 1] == 1 and cappredictiony_predhA [k] == 0 and cappredictiony_predhA [k + 1] == 1:
                cappredictiony_predhA [k] = 1  
                
                
     
                
     
    # cappredictiony_predhN=objectNRT[0]
    # for k in range (len(cappredictiony_predhN)): # decrese the missclassification by converting the classified A phases into not-A when the NREM classifiers indicates a period of REM/wake (A phase can only occur during NREM sleep)
    #     if cappredictiony_predhN[k]==0:
    #         YTestOneLine[k]=0
    # for k in range (len (cappredictiony_predhA) - 1):
    #     if k > 0:
    #         if cappredictiony_predhA [k - 1] == 0 and cappredictiony_predhA [k] == 1 and cappredictiony_predhA [k + 1] == 0:
    #             YTestOneLine [k] = 0
                
    # for k in range (len (cappredictiony_predhA)-1):
    #     if k > 0:
    #         if cappredictiony_predhA [k - 1] == 1 and cappredictiony_predhA [k] == 0 and cappredictiony_predhA [k + 1] == 1:
    #             YTestOneLine [k] = 1  
                
                
                
                
    print ("\n\n Testing A phase after ensemble for subject ", ee, ", epoch ", ff)    
    tn, fp, fn, tp = confusion_matrix (YTestOneLine, cappredictiony_predhA, labels=[0,1]).ravel()
    print (classification_report (YTestOneLine, cappredictiony_predhA))
    accuracy0 = (tp + tn) / (tp + tn + fp + fn)
    print ('Accuracy : ', accuracy0)
    sensitivity0 = tp / (tp + fn)
    print ('Sensitivity : ', sensitivity0)
    specificity0 = tn / (fp + tn)
    print ('Specificity : ', specificity0)
    
    AccAtInterAC [ff] = accuracy0
    SenAtInterAC [ff] = sensitivity0
    SpeAtInterAC [ff] = specificity0
    TPInterAC [ff] = tp
    TNInterAC [ff] = tn
    FPInterAC [ff] = fp
    FNInterAC [ff] = fn   
    PPVInterAC [ff] = tp / (tp + fp)
    NPVInterAC [ff] = tn / (tn + fn)
    AtotalInter [ff] = sum (cappredictiony_predhA) # total number of epochs classified as A phase      
    
    
    AccAtEndAC [ee] = np.mean (AccAtInterAC)
    SenAtEndAC [ee] = np.mean (SenAtInterAC)
    SpeAtEndAC [ee] = np.mean (SpeAtInterAC)
    TPEndAC [ee] = np.mean (TPInterAC)
    TNEndAC [ee] = np.mean (TNInterAC)
    FPEndAC [ee] = np.mean (FPInterAC)
    FNEndAC [ee] = np.mean (FNInterAC)
    PPVEndAC [ee] = np.mean (PPVInterAC)
    NPVEndAC [ee] = np.mean (NPVInterAC)    
    AtotalEnd [ee] = np.mean (AtotalInter)
    
    
    metricsAC = np.c_ [AccAtEndAC, SenAtEndAC, SpeAtEndAC, TPEndAC, TNEndAC, FPEndAC, FNEndAC, PPVEndAC, NPVEndAC, AtotalEnd] # variable holding all the results for the A phase analysis after ensemble
    
    # if useSDpatients > 0:
    #     StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
    #     f = open (StringText, 'ab')
    #     pickle.dump (cappredictiony_predhA, f)
    #     f.close ()
    # else:
    #     StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_PostProcessing_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
    #     f = open (StringText, 'ab')
    #     pickle.dump (cappredictiony_predhA, f)
    #     f.close ()
    
    
    
    
    # m = ConfusionMatrix (actual_vector = YTestOneLine, predict_vector = cappredictiony_predhA)
    # Acc_class = m.Overall_ACC
    # Sen_class = m.TPR
    # Spe_class = m.TNR
    # AUC_class = m.AUC
    # PPV_class = m.PPV
    # NPV_class = m.NPV
    # print("Acc : " + str(Acc_class) + "\n")
    # print("Sen : " + str(Sen_class) + "\n")
    # print("Spe : " + str(Spe_class) + "\n")
    # print("AUC : " + str(AUC_class) + "\n")
    # print("PPV : " + str(PPV_class) + "\n")
    # print("NPV : " + str(NPV_class) + "\n")
    
    