"""
##############################################################################
#
# model developed to estimate the A phase, NREM sleep, CAP cycles, and CAP 
# rate, the model uses one enseble of three CNN to predict the A phase and 
# another enseble of three CNN to predict the NREM, each classifier of the
# enseble is fed with a diffen overlapping window from the EEG signal, the 
# CAP cycles are estimated using a finite state machine that is fed with the 
# classified A phases, each ecpoch is composed on 100 sample points (signal 
# sampled at 100 Hz) and the model is performing a second by second analysis
#
##############################################################################
"""
"""
# import lybraries
"""
import scipy.io as spio # to load the .mat files with the data
import numpy as np # for mathematical notation and examination
import tensorflow as tf # for clearing the secction, realeasing the GPU memory after a training cycle 
from tensorflow.keras.utils import to_categorical # convert the data labels to categorical labels, required for training and testing the models
from tensorflow.keras.models import Sequential # to use a sequential topology for the structure of the classifier
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D # layers used by the developed classifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report # to perform the examination of the results
from sklearn.utils import class_weight # to use cost sensitive learning
from sklearn.model_selection import train_test_split # to crate the validation dataset from the training dataset
import gc # to release the holded memory in the garbage collector
import pickle # to store the results of the examination in files
import os # to change the working directory
"""
# Control variables
"""
numberSubjectsN = 14 # number of normal subjects to be considered (from 0 to numberSubjectsN)
numberSubjectsSD = 3 # number of subjects with sleep Disorder to be considered (from 0 to numberSubjectsSD)
startEpochs = 0 # 4 number of the subject to start the leave one out examination
useSDpatients = 0 # 0 to use only healthy, 1 to use healthy and SDB, 2 to use healthy and NFLE, 3 to use PLM, X to use ins
Begin = 0 # location of the sorted array where the data for the first subject used to compose the training dataset for the leave one out examination is identified, the training dataset is composed of subejcts from Begin to BeginTest
EpochsWork = 1 # number of examined iterations for each subejct
OverLap = [8, 12, 8] # number of overlapping seconds to be considered in the overlapping windows of A phase analysis (need to be 0 or an odd number), either [0] if no overlapping or [amount of overlapping for right, amount of overlapping for center, amount of overlapping for left] to use overlapping
OverLapH = [16, 14, 16] # number of overlapping seconds to be considered in the overlapping windows of NREM analysis (need to be 0 or an odd number), either [0] if no overlapping or [amount of overlapping for right, amount of overlapping for center, amount of overlapping for left] to use overlapping
thresholdEarlyStoping = 0.005 # minimum increasse in the Area Under the receiving operating Curve (AUC) considered by the early stopping procedure
patienteceValue = 10 # value used by the early stopping procedure to specify the number of training cycles without a minimum increasse of thresholdEarlyStoping befor stoping the examination
KernelSize = 5 # size of the kernel used by the convolution layer
APhaseSubtype = 1 # 0 for A and not-A; 1 for A1 and not-A1; 2 for A2 and not-A2; 3 for A3 and not-A3
os.chdir("D:\Github\EnsembleCNN\Data") # change the working directory
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
# for A phase examination of all cycles, up to EpochsWork, for each subejct
AccAtInterA = np.zeros ([EpochsWork, 3]) # variable holding the accuracy for the A phase estimation of each classifier composing the classifier ensemble
SenAtInterA = np.zeros ([EpochsWork, 3]) # variable holding the sensitivity for the A phase estimation of each classifier composing the classifier ensemble
SpeAtInterA = np.zeros ([EpochsWork, 3]) # variable holding the specificity for the A phase estimation of each classifier composing the classifier ensemble
AUCAtInterA = np.zeros ([EpochsWork, 3]) # variable holding the AUC for the A phase estimation of each classifier composing the classifier ensemble
TPInterA = np.zeros ([EpochsWork, 3]) # variable holding the true positives for the A phase estimation of each classifier composing the classifier ensemble
TNInterA = np.zeros ([EpochsWork, 3]) # variable holding the true negatives for the A phase estimation of each classifier composing the classifier ensemble
FPInterA = np.zeros ([EpochsWork, 3]) # variable holding the false positives for the A phase estimation of each classifier composing the classifier ensemble
FNInterA = np.zeros ([EpochsWork, 3]) # variable holding the false negatives for the A phase estimation of each classifier composing the classifier ensemble
PPVInterA = np.zeros ([EpochsWork, 3]) # variable holding the positive predictive value for the A phase estimation of each classifier composing the classifier ensemble
NPVInterA = np.zeros ([EpochsWork, 3]) # variable holding the negative predictive value for the A phase estimation of each classifier composing the classifier ensemble
# for A phase examination of all cycles, up to EpochsWork, for each subejct after the classifier ensemble
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
# for NREM examination of all cycles, up to EpochsWork, for each subejct
AccAtInterN = np.zeros ([EpochsWork, 3]) # variable holding the accuracy for the NREM estimation of each classifier composing the classifier ensemble
SenAtInterN = np.zeros ([EpochsWork, 3]) # variable holding the sensitivity for the NREM estimation of each classifier composing the classifier ensemble
SpeAtInterN = np.zeros ([EpochsWork, 3]) # variable holding the specificity for the NREM estimation of each classifier composing the classifier ensemble
AUCAtInterN = np.zeros ([EpochsWork, 3]) # variable holding the AUC for the NREM estimation of each classifier composing the classifier ensemble
TPInterN = np.zeros ([EpochsWork, 3]) # variable holding the true positives for the NREM estimation of each classifier composing the classifier ensemble
TNInterN = np.zeros ([EpochsWork, 3]) # variable holding the true negatives for the NREM estimation of each classifier composing the classifier ensemble
FPInterN = np.zeros ([EpochsWork, 3]) # variable holding the false positives for the NREM estimation of each classifier composing the classifier ensemble
FNInterN = np.zeros ([EpochsWork, 3]) # variable holding the false negatives for the NREM estimation of each classifier composing the classifier ensemble
PPVInterN = np.zeros ([EpochsWork, 3]) # variable holding the positive predictive value for the NREM estimation of each classifier composing the classifier ensemble
NPVInterN = np.zeros ([EpochsWork, 3]) # variable holding the negative predictive value for the NREM estimation of each classifier composing the classifier ensemble
# for NREM examination of all cycles, up to EpochsWork, for each subejct after the classifier ensemble
AccAtInterNC = np.zeros (EpochsWork) # variable holding the accuracy for the NREM estimation after the classifier ensemble
SenAtInterNC = np.zeros (EpochsWork) # variable holding the sensitivity for the NREM estimation after the classifier ensemble
SpeAtInterNC = np.zeros (EpochsWork) # variable holding the specificity for the NREM estimation after the classifier ensemble
TPInterNC = np.zeros (EpochsWork) # variable holding the true positives for the NREM estimation after the classifier ensemble
TNInterNC = np.zeros (EpochsWork) # variable holding the true negatives for the NREM estimation after the classifier ensemble
FPInterNC = np.zeros (EpochsWork) # variable holding the false positives for the NREM estimation after the classifier ensemble
FNInterNC = np.zeros (EpochsWork) # variable holding the false negatives for the NREM estimation after the classifier ensemble
PPVInterNC = np.zeros (EpochsWork) # variable holding the positive predictive value for the NREM estimation after the classifier ensemble
NPVInterNC = np.zeros (EpochsWork) # variable holding the negative predictive value for the NREM estimation after the classifier ensemble
NREMtotalInter = np.zeros (EpochsWork) # variable holding the predicted duration of all NREM epochs
# for CAP cycle examination of all cycles, up to EpochsWork, for each subejct
AccAtInter = np.zeros (EpochsWork) # variable holding the accuracy for the CAP cycle estimation 
SenAtInter = np.zeros (EpochsWork) # variable holding the sensitivity for the CAP cycle estimation 
SpeAtInter = np.zeros (EpochsWork) # variable holding the specificity for the CAP cycle estimation 
AUCAtInter = np.zeros (EpochsWork) # variable holding the AUC for the CAP cycle estimation 
TPInter = np.zeros (EpochsWork) # variable holding the true positives for the CAP cycle estimation 
TNInter = np.zeros (EpochsWork) # variable holding the true negatives for the CAP cycle estimation 
FPInter = np.zeros (EpochsWork) # variable holding the false positives for the CAP cycle estimation 
FNInter = np.zeros (EpochsWork) # variable holding the false negatives for the CAP cycle estimation 
PPVInter = np.zeros (EpochsWork) # variable holding the positive predictive value for the CAP cycle estimation 
NPVInter = np.zeros (EpochsWork) # variable holding the negative predictive value for the CAP cycle estimation 
CAPtotalInter = np.zeros (EpochsWork) # variable holding the predicted duration of all CAP cycle epochs
# for CAP rate examination of all cycles, up to EpochsWork, for each subejct
CAPrateErrototalInter = np.zeros (EpochsWork) # variable holding the predicted CAP rate error
CAPrateErroPercentagetotalInter = np.zeros (EpochsWork) # variable holding the predicted CAP rate error percentage
# for A phase examination of all subects cosidered for the leave one out procedure
AccAtEndA = np.zeros ([Epochs, 3]) # variable holding the accuracy for the A phase estimation of each subejcts evaluated by the leave one out procedure
SenAtEndA = np.zeros ([Epochs, 3]) # variable holding the sensitivity for the A phase estimation of each subejcts evaluated by the leave one out procedure 
SpeAtEndA = np.zeros ([Epochs, 3]) # variable holding the specifiity for the A phase estimation of each subejcts evaluated by the leave one out procedure
AUCAtEndA = np.zeros ([Epochs, 3]) # variable holding the AUC for the A phase estimation of each subejcts evaluated by the leave one out procedure
TPEndA = np.zeros ([Epochs, 3]) # variable holding the true positives for the A phase estimation of each subejcts evaluated by the leave one out procedure
TNEndA = np.zeros ([Epochs, 3]) # variable holding the true negatives for the A phase estimation of each subejcts evaluated by the leave one out procedure
FPEndA = np.zeros ([Epochs, 3]) # variable holding the false positives for the A phase estimation of each subejcts evaluated by the leave one out procedure
FNEndA = np.zeros ([Epochs, 3]) # variable holding the false negatives for the A phase estimation of each subejcts evaluated by the leave one out procedure
PPVEndA = np.zeros ([Epochs, 3]) # variable holding the positive predictive value for the A phase estimation of each subejcts evaluated by the leave one out procedure
NPVEndA = np.zeros ([Epochs, 3]) # variable holding the negative predictive value for the A phase estimation of each subejcts evaluated by the leave one out procedure
# for NREM examination of all subects cosidered for the leave one out procedure
AccAtEndN = np.zeros ([Epochs, 3]) # variable holding the accuracy for the NREM estimation of each subejcts evaluated by the leave one out procedure
SenAtEndN = np.zeros ([Epochs, 3]) # variable holding the sensitivity for the NREM estimation of each subejcts evaluated by the leave one out procedure 
SpeAtEndN = np.zeros ([Epochs, 3]) # variable holding the specifiity for the NREM estimation of each subejcts evaluated by the leave one out procedure
AUCAtEndN = np.zeros ([Epochs, 3]) # variable holding the AUC for the NREM estimation of each subejcts evaluated by the leave one out procedure
TPEndN = np.zeros ([Epochs, 3]) # variable holding the true positives for the NREM estimation of each subejcts evaluated by the leave one out procedure
TNEndN = np.zeros ([Epochs, 3]) # variable holding the true negatives for the NREM estimation of each subejcts evaluated by the leave one out procedure
FPEndN = np.zeros ([Epochs, 3]) # variable holding the false positives for the NREM estimation of each subejcts evaluated by the leave one out procedure
FNEndN = np.zeros ([Epochs, 3]) # variable holding the false negatives for the NREM estimation of each subejcts evaluated by the leave one out procedure
PPVEndN = np.zeros ([Epochs, 3]) # variable holding the positive predictive value for the NREM estimation of each subejcts evaluated by the leave one out procedure
NPVEndN = np.zeros ([Epochs, 3]) # variable holding the negative predictive value for the NREM estimation of each subejcts evaluated by the leave one out procedure
# for A phase examination of all subects cosidered for the leave one out procedure after the classifier ensemble
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
AtotalEnd = np.zeros(Epochs) # variable holding the predicted duration of all A phases epochs of each subejcts after the classifier ensemble
# for NREM examination of all subects cosidered for the leave one out procedure after the classifier ensemble
AccAtEndNC = np.zeros (Epochs) # variable holding the accuracy for the NREM estimation of each subejcts after the classifier ensemble
SenAtEndNC = np.zeros (Epochs) # variable holding the sensitivity for the NREM estimation of each subejcts after the classifier ensemble 
SpeAtEndNC = np.zeros (Epochs) # variable holding the specifiity for the NREM estimation of each subejcts after the classifier ensemble
AUCAtEndNC = np.zeros (Epochs) # variable holding the AUC for the NREM estimation of each subejcts after the classifier ensemble
TPEndNC = np.zeros (Epochs) # variable holding the true positives for the NREM estimation of each subejcts after the classifier ensemble
TNEndNC = np.zeros (Epochs) # variable holding the true negatives for the NREM estimation of each subejcts after the classifier ensemble
FPEndNC = np.zeros (Epochs) # variable holding the false positives for the NREM estimation of each subejcts after the classifier ensemble
FNEndNC = np.zeros (Epochs) # variable holding the false negatives for the NREM estimation of each subejcts after the classifier ensemble
PPVEndNC = np.zeros (Epochs) # variable holding the positive predictive value for the NREM estimation of each subejcts after the classifier ensemble
NPVEndNC = np.zeros (Epochs) # variable holding the negative predictive value for the NREM estimation of each subejcts after the classifier ensemble
NREMtotalEnd = np.zeros(Epochs) # variable holding the predicted duration of all NREM epochs of each subejcts after the classifier ensemble
# for CAP cycle examination of all subects cosidered for the leave one out procedure
AccAtEnd = np.zeros (Epochs) # variable holding the accuracy for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
SenAtEnd = np.zeros (Epochs) # variable holding the sensitivity for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure 
SpeAtEnd = np.zeros (Epochs) # variable holding the specifiity for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
AUCAtEnd = np.zeros (Epochs) # variable holding the AUC for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
TPEnd = np.zeros (Epochs) # variable holding the true positives for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
TNEnd = np.zeros (Epochs) # variable holding the true negatives for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
FPEnd = np.zeros (Epochs) # variable holding the false positives for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
FNEnd = np.zeros (Epochs) # variable holding the false negatives for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
PPVEnd = np.zeros (Epochs) # variable holding the positive predictive value for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
NPVEnd = np.zeros (Epochs) # variable holding the negative predictive value for the CAP cycle estimation of each subejcts evaluated by the leave one out procedure
CAPtotalEnd = np.zeros(Epochs) # variable holding the predicted duration of all CAP cycles epochs of each subejcts evaluated by the leave one out procedure
# for CAP rate examination of all subects cosidered for the leave one out procedure
CAPrateErrototalEnd = np.zeros (Epochs) # variable holding the predicted CAP rate error for all subjects
CAPrateErroPercentagetotalEnd = np.zeros (Epochs) # variable holding the predicted CAP rate error percentage for all subjects
"""
# leave one out examination
"""
for ee in range (startEpochs, Epochs, 1): # examine from subejct identified by startEpochsup to the subejct identified by Epochs
    print ('\n\n Subject: ', ee)
    for ff in range (EpochsWork): # repeate the simulation EpochsWork times (number of examine cycles) to attain statistically significant results
        tf.keras.backend.clear_session()
        print ('\n\n Epoch Work: ', ff)
        # for A phase
        if len (OverLap) == 1: # determine the total duration of overlap
            OverlappingCenter = 0 # model without overlapping
            overlapingSide = [1] # if the model should not consider overlapping windows, then only test one classifier with leave one out results, leaving the results for the other two classifiers composed of only zeros 
        else: # model with overlapping
            OverlappingRight = OverLap [0] - 1 # total duration of overlapp to the right, for the A phase analyis, without the epoch related to the label
            OverlappingCenter = OverLap [1] - 1 # total duration of overlapp to the right and left, for the A phase analyis, without the epoch related to the label
            OverlappingLeft = OverLap [2] - 1 # total duration of overlapp to the left, for the A phase analyis, without the epoch related to the label
            overlapingSide = [0, 1, 2] # if the model should consider overlapping windows, then test the thre considered overlapping scenarios
        # for NREM
        if len (OverLapH) == 1:
            OverlappingCenterH = 0 
            overlapingSideH = [1]
        else:
            OverlappingRightH = OverLapH [0] - 1 
            OverlappingCenterH = OverLapH [1] - 1
            OverlappingLeftH = OverLapH [2] - 1
            overlapingSideH = [0, 1, 2]
        """
        # create the overlappign data
        """
        for a in range (0, len(overlapingSide), 1): # test the three overllaping scenarios if overlapping should be considered or only one examine one classifier if overlapping is not considered (OverLap = 0)
            """    
            # load the database
            """
            for JJ in range (numberSubjectsN + 1): # load the normal subjects' data
                if JJ < 11: # from subject 0 to 10
                    Datadata = "n" + str (JJ + 1) + "eegminut2.mat" # string with the subejct data
                    labName = "n" + str (JJ + 1) + "eegminutLable2.mat" # string with the subejct A phase labels
                    labNameh = "n" + str (JJ + 1) + "hypnoEEGminutLable2V2.mat" # string with the subejct NREM labels
                    mat = spio.loadmat (Datadata, squeeze_me = True) # load the subject's data
                    Datadata = mat.get ('eegSensor') # dictionary holding the subject's data
                    del mat # delet the dictionary with the subject's data
                    Datadata = (Datadata - np.mean (Datadata)) / np.std (Datadata) # standardize the data        
                    mat = spio.loadmat (labName, squeeze_me = True) # load the subject's A phase labels
                    labName = mat.get ('CAPlabel1') # dictionary holding the subject's A phase labels
                    del mat # delet the dictionary with the subject's data
                    mat = spio.loadmat (labNameh, squeeze_me = True) # load the subject's NREM labels
                    labNameh = mat.get ('Hipno') # dictionary holding the subject's NREM labels
                    del mat # delet the dictionary with the subject's data
                    labNameh [labNameh == 5] = 0 # convert the REM labels to 0, refering both REM and wake periods, desinating the not-NREM data
                    labNameh [labNameh > 0] = 1 # convert the sleep stage labels into only NREM labels
                    if JJ == 0: # subejct 0
                        n1 = Datadata # varaible holding heathy subject's 0 data
                        nc1 = labName # varaible holding subject's 0 A phase labels
                        nch1 = labNameh # varaible holding subject's 0 NREM labels
                    elif JJ == 1: # subejct 1
                        n2 = Datadata # varaible holding heathy subject's 1 data
                        nc2 = labName # varaible holding subject's 1 A phase labels
                        nch2 = labNameh # varaible holding subject's 1 NREM labels
                    elif JJ == 2: # subejct 2
                        n3 = Datadata # varaible holding heathy subject's 2 data
                        nc3 = labName # varaible holding subject's 2 A phase labels
                        nch3 = labNameh # varaible holding subject's 2 NREM labels
                    elif JJ == 3: # subejct 3
                        n4 = Datadata # varaible holding heathy subject's 3 data
                        nc4 = labName # varaible holding subject's 3 A phase labels
                        nch4 = labNameh # varaible holding subject's 3 NREM labels
                    elif JJ == 4: # subejct 4
                        n5 = Datadata # varaible holding heathy subject's 4 data
                        nc5 = labName # varaible holding subject's 4 A phase labels
                        nch5 = labNameh # varaible holding subject's 4 NREM labels
                    elif JJ == 5: # subejct 5
                        n6 = Datadata # varaible holding heathy subject's 5 data
                        nc6 = labName # varaible holding subject's 5 A phase labels
                        nch6 = labNameh # varaible holding subject's 5 NREM labels
                    elif JJ == 6: # subejct 6
                        n7 = Datadata # varaible holding heathy subject's 6 data
                        nc7 = labName # varaible holding subject's 6 A phase labels
                        nch7 = labNameh # varaible holding subject's 6 NREM labels
                    elif JJ == 7: # subejct 7
                        n8 = Datadata # varaible holding heathy subject's 7 data
                        nc8 = labName # varaible holding subject's 7 A phase labels
                        nch8 = labNameh # varaible holding subject's 7 NREM labels
                    elif JJ == 8: # subejct 8
                        n9 = Datadata # varaible holding heathy subject's 8 data
                        nc9 = labName # varaible holding subject's 8 A phase labels
                        nch9 = labNameh # varaible holding subject's 8 NREM labels
                    elif JJ == 9: # subejct 9
                        n10 = Datadata # varaible holding heathy subject's 9 data
                        nc10 = labName # varaible holding subject's 9 A phase labels
                        nch10 = labNameh # varaible holding subject's 9 NREM labels
                    else: # subejct 10
                        n11 = Datadata # varaible holding heathy subject's 10 data
                        nc11 = labName # varaible holding subject's 10 A phase labels
                        nch11 = labNameh # varaible holding subject's 10 NREM labels
                else: # from subject 11 to 14
                    Datadata = "n" + str (JJ + 2) + "eegminut2.mat"
                    labName = "n" + str (JJ + 2) + "eegminutLable2.mat"
                    labNameh = "n" + str (JJ + 2) + "hypnoEEGminutLable2V2.mat"
                    mat = spio.loadmat (Datadata, squeeze_me = True)
                    Datadata = mat.get ('eegSensor')
                    del mat
                    Datadata = (Datadata - np.mean (Datadata)) / np.std (Datadata)       
                    mat = spio.loadmat (labName, squeeze_me = True)
                    labName = mat.get ('CAPlabel1')
                    del mat
                    mat = spio.loadmat (labNameh, squeeze_me = True)
                    labNameh = mat.get ('Hipno')
                    del mat
                    labNameh[labNameh == 5] = 0
                    labNameh[labNameh > 0] = 1
                    if JJ == 11: # subejct 11
                        n13 = Datadata # varaible holding heathy subject's 11 data
                        nc13 = labName # varaible holding subject's 11 A phase labels
                        nch13 = labNameh # varaible holding subject's 11 NREM labels
                    elif JJ == 12: # subejct 12
                        n14 = Datadata # varaible holding heathy subject's 12 data    
                        nc14 = labName # varaible holding subject's 12 A phase labels
                        nch14 = labNameh # varaible holding subject's 12 NREM labels
                    elif JJ == 13: # subejct 13
                        n15 = Datadata # varaible holding heathy subject's 13 data
                        nc15 = labName # varaible holding subject's 13 A phase labels
                        nch15 = labNameh # varaible holding subject's 13 NREM labels
                    else: # subejct 14
                        n16 = Datadata # varaible holding heathy subject's 14 data
                        nc16 = labName # varaible holding subject's 14 A phase labels
                        nch16 = labNameh # varaible holding subject's 14 NREM labels
            if useSDpatients > 0:
                for JJ in range (numberSubjectsSD + 1): # load the SD subjects' data
                    if useSDpatients == 1: # for SDB
                        KK = JJ
                    elif useSDpatients == 2: # for nfle
                        if JJ == 0: 
                            KK = 6
                        elif JJ == 1: 
                            KK = 12
                        elif JJ == 2: 
                            KK = 13
                        else:
                            KK = 14
                    elif useSDpatients == 3: # for plm
                        if JJ == 0: 
                            KK = 1
                        elif JJ == 1: 
                            KK = 2
                        elif JJ == 2: 
                            KK = 7
                        else:
                            KK = 9
                    else: # for ins
                        if JJ == 0: 
                            KK = 1
                        elif JJ == 1: 
                            KK = 4
                        elif JJ == 2: 
                            KK = 6
                        else:
                            KK = 8
                    if useSDpatients <= 2:
                        Datadata = str (disorder) + str (KK + 1) + "eegminut2.mat"
                    else:
                        Datadata = str (disorder) + str (KK + 1) + "C4V2.mat"
                    labName = str (disorder) + str (KK + 1) + "eegminutLable2.mat"
                    labNameh = str (disorder) + str (KK + 1) + "hypnoEEGminutLable2V2.mat"
                    mat = spio.loadmat (Datadata, squeeze_me = True)
                    if useSDpatients == 1 or useSDpatients == 0:
                        Datadata = mat.get ('eegSensor')
                    else:
                        Datadata = mat.get ('c4')
                    del mat
                    Datadata = (Datadata - np.mean (Datadata)) / np.std (Datadata)       
                    mat = spio.loadmat (labName, squeeze_me = True)
                    labName = mat.get ('CAPlabel1')
                    del mat
                    mat = spio.loadmat (labNameh, squeeze_me = True)
                    labNameh = mat.get ('Hipno')
                    del mat
                    labNameh[labNameh == 5] = 0
                    labNameh[labNameh > 0] = 1
                    if JJ == 0: # subejct SD 0
                        sd1 = Datadata # varaible holding SD subject's 0 data
                        sdc1 = labName # varaible holding SD subject's 0 A phase labels
                        sdch1 = labNameh # varaible holding SD subject's 0 NREM labels
                    elif JJ == 1: # subejct SD 1
                        sd2 = Datadata # varaible holding SD subject's 1 data
                        sdc2 = labName # varaible holding SD subject's 1 phase labels
                        sdch2 = labNameh # varaible holding SD subject's 1 NREM labels
                    elif JJ == 2: # subejct SD 2
                        sd3 = Datadata # varaible holding SD subject's 2 data
                        sdc3 = labName # varaible holding SD subject's 2 A phase labels
                        sdch3 = labNameh # varaible holding SD subject's 2 NREM labels
                    else: # subejct SD 3
                        sd4 = Datadata # varaible holding SD subject's 3 data
                        sdc4 = labName # varaible holding SD subject's 3 A phase labels
                        sdch4 = labNameh # varaible holding SD subject's 3 NREM labels      
            """
            # produce the overlapping scenarios for A phase
            """
            if overlapingSide [a] == 0: # test overlapping to the right: first 100 points refer to the epoch's label and the remaining poins are overlapping to the right  
                features = 100 * (OverlappingRight * 2 + 1) # number of features fed to the classifier at each epoch
                for k in range (numberSubjectsN + 1): # examined normal subjects
                    if k < 11: # from subject 0 to 10
                        dataName = "n" + str (k + 1) # select the subejct's data to produce the overlapping
                        Datadata = eval (dataName) # select the variable holding the subject's data
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRight * 2), OverlappingRight * 2 * 100 + 100)) # variable that will hold the reshaped subject's data
                        counting = 0 # conting variable to hold the number of evaluated epochs
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRight * 2 - OverlappingRight)), 1): # produce the OverlappingRight data for each epoch wheer it is possible to produce data
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRight * 100 * 2] # copy the OverlappingRight data to the variable
                            counting = counting + 1 # increase the number of examined subejcts
                        if k == 0: # subejct 0
                            nA1 = DatadataV2 
                            nc1 = nc1 [0 : len (nc1) - OverlappingRight * 2] # remove the A phase labels related to the epochs were it is not possible to produce the intended OverlappingRight window
                        elif k == 1: # subejct 1
                            nA2 = DatadataV2
                            nc2 = nc2 [0 : len (nc2) - OverlappingRight * 2]
                        elif k == 2: # subejct 2
                            nA3 = DatadataV2
                            nc3 = nc3 [0 : len (nc3) - OverlappingRight * 2]
                        elif k == 3: # subejct 3
                            nA4 = DatadataV2
                            nc4 = nc4 [0 : len (nc4) - OverlappingRight * 2]
                        elif k == 4: # subejct 4
                            nA5 = DatadataV2
                            nc5 = nc5 [0 : len (nc5) - OverlappingRight * 2]
                        elif k == 5: # subejct 5
                            nA6 = DatadataV2
                            nc6 = nc6 [0 : len (nc6) - OverlappingRight * 2]
                        elif k == 6: # subejct 6
                            nA7 = DatadataV2
                            nc7 = nc7 [0 : len (nc7) - OverlappingRight * 2]
                        elif k == 7: # subejct 7
                            nA8 = DatadataV2
                            nc8 = nc8 [0 : len (nc8) - OverlappingRight * 2] 
                        elif k == 8: # subejct 8
                            nA9 = DatadataV2
                            nc9 = nc9 [0 : len (nc9) - OverlappingRight * 2]
                        elif k == 9: # subejct 9
                            nA10 = DatadataV2
                            nc10 = nc10 [0 : len (nc10) - OverlappingRight * 2]
                        else: # subejct 10
                            nA11 = DatadataV2
                            nc11 = nc11 [0 : len (nc11) - OverlappingRight * 2]
                    else:
                        dataName = "n" + str (k + 2)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRight * 2), OverlappingRight * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRight * 2 - OverlappingRight)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRight * 100 * 2]
                            counting = counting + 1
                        if k == 11:
                            nA13 = DatadataV2
                            nc13 = nc13 [0 : len (nc13) - OverlappingRight * 2]
                        elif k == 12:
                            nA14=DatadataV2
                            nc14 = nc14 [0 : len (nc14) - OverlappingRight * 2]
                        elif k == 13:
                            nA15 = DatadataV2
                            nc15 = nc15 [0 : len (nc15) - OverlappingRight * 2]
                        else:
                            nA16 = DatadataV2
                            nc16 = nc16 [0 : len (nc16) - OverlappingRight * 2]
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRight * 2), OverlappingRight * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRight * 2 - OverlappingRight)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRight * 100 * 2]
                            counting = counting + 1
                        if k == 0:
                            sdbA1 = DatadataV2
                            sdc1 = sdc1 [0 : len (sdc1) - OverlappingRight * 2]
                        elif k == 1:
                            sdbA2 = DatadataV2
                            sdc2 = sdc2 [0 : len (sdc2) - OverlappingRight * 2]
                        elif k == 2:
                            sdbA3 = DatadataV2
                            sdc3 = sdc3 [0 : len (sdc3) - OverlappingRight * 2]
                        else:
                            sdbA4 = DatadataV2
                            sdc4 = sdc4 [0 : len (sdc4) - OverlappingRight * 2]
            elif overlapingSide [a] == 1: # test overlapping to the left and right: central 100 points refer to the epoch's label and the remaining poins are overlapping to either left or right    
                features = 100 * (OverlappingCenter * 2 + 1)
                for k in range (numberSubjectsN + 1):
                    if k < 11:
                        dataName = "n" + str (k + 1) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenter * 2), OverlappingCenter * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenter, int ((len (Datadata) / 100 - OverlappingCenter * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenter * 100 : (x * 100 + 100) + OverlappingCenter * 100]
                            counting = counting + 1
                        if k == 0:
                            nA1 = DatadataV2
                            nc1 = nc1 [OverlappingCenter : len (nc1) - OverlappingCenter]
                        elif k == 1:
                            nA2 = DatadataV2
                            nc2 = nc2 [OverlappingCenter : len (nc2) - OverlappingCenter]
                        elif k == 2:
                            nA3 = DatadataV2
                            nc3 = nc3 [OverlappingCenter : len (nc3) - OverlappingCenter]
                        elif k == 3:
                            nA4 = DatadataV2
                            nc4 = nc4 [OverlappingCenter : len (nc4) - OverlappingCenter]
                        elif k == 4:
                            nA5 = DatadataV2
                            nc5 = nc5 [OverlappingCenter : len (nc5) - OverlappingCenter]
                        elif k == 5:
                            nA6 = DatadataV2
                            nc6 = nc6 [OverlappingCenter : len (nc6) - OverlappingCenter]
                        elif k == 6:
                            nA7 = DatadataV2
                            nc7 = nc7 [OverlappingCenter : len (nc7) - OverlappingCenter]
                        elif k == 7:
                            nA8 = DatadataV2
                            nc8 = nc8 [OverlappingCenter : len (nc8) - OverlappingCenter]
                        elif k == 8:
                            nA9 = DatadataV2
                            nc9 = nc9 [OverlappingCenter : len (nc9) - OverlappingCenter]
                        elif k == 9:
                            nA10 = DatadataV2
                            nc10 = nc10 [OverlappingCenter : len (nc10) - OverlappingCenter]
                        else:
                            nA11 = DatadataV2
                            nc11 = nc11 [OverlappingCenter : len (nc11) - OverlappingCenter]
                    else:
                        dataName = "n" + str (k + 2)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenter * 2), OverlappingCenter * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenter, int ((len (Datadata) / 100 - OverlappingCenter * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenter * 100 : (x * 100 + 100) + OverlappingCenter * 100]
                            counting = counting + 1
                        if k == 11:
                            nA13 = DatadataV2
                            nc13 = nc13 [OverlappingCenter : len (nc13) - OverlappingCenter]
                        elif k == 12:
                            nA14 = DatadataV2
                            nc14 = nc14 [OverlappingCenter : len (nc14) - OverlappingCenter]
                        elif k == 13:
                            nA15 = DatadataV2
                            nc15 = nc15 [OverlappingCenter : len (nc15) - OverlappingCenter]
                        else:
                            nA16 = DatadataV2
                            nc16 = nc16 [OverlappingCenter : len (nc16) - OverlappingCenter]
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenter * 2), OverlappingCenter * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenter, int ((len (Datadata) / 100 - OverlappingCenter * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenter * 100 : (x * 100 + 100) + OverlappingCenter * 100]
                            counting = counting + 1
                        if k == 0:
                            sdbA1 = DatadataV2
                            sdc1 = sdc1 [OverlappingCenter : len (sdc1) - OverlappingCenter]
                        elif k == 1:
                            sdbA2 = DatadataV2
                            sdc2 = sdc2 [OverlappingCenter : len (sdc2) - OverlappingCenter]
                        elif k == 2:
                            sdbA3 = DatadataV2
                            sdc3 = sdc3 [OverlappingCenter : len (sdc3) - OverlappingCenter]
                        else:
                            sdbA4 = DatadataV2
                            sdc4 = sdc4 [OverlappingCenter : len (sdc4) - OverlappingCenter]
            else: # test overlapping to the left: last 100 points refer to the epoch's label and the remaining poins are overlapping to the left
                features = 100 * (OverlappingLeft * 2 + 1)
                for k in range (numberSubjectsN + 1):
                    if k < 11:
                        dataName = "n" + str (k + 1) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeft * 2), OverlappingLeft * 2 * 100 + 100))
                        counting = 0
                        for x in range(0, int((len(Datadata)/100-OverlappingLeft*2-OverlappingLeft)), 1):
                            DatadataV2 [counting, ]= Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeft * 100 * 2]
                            counting = counting + 1
                        if k == 0:
                            nA1 = DatadataV2
                            nc1 = nc1 [OverlappingLeft * 2 : len (nc1)]
                        elif k == 1:
                            nA2 = DatadataV2
                            nc2 = nc2 [OverlappingLeft * 2 : len (nc2)]
                        elif k == 2:
                            nA3 = DatadataV2
                            nc3 = nc3 [OverlappingLeft * 2 : len (nc3)]
                        elif k == 3:
                            nA4 = DatadataV2
                            nc4 = nc4 [OverlappingLeft * 2 : len (nc4)]
                        elif k == 4:
                            nA5 = DatadataV2
                            nc5 = nc5 [OverlappingLeft * 2 : len (nc5)]
                        elif k == 5:
                            nA6 = DatadataV2
                            nc6 = nc6 [OverlappingLeft * 2 : len (nc6)]
                        elif k == 6:
                            nA7 = DatadataV2
                            nc7 = nc7 [OverlappingLeft * 2 : len (nc7)]
                        elif k == 7:
                            nA8 = DatadataV2
                            nc8 = nc8 [OverlappingLeft * 2 : len (nc8)]
                        elif k == 8:
                            nA9 = DatadataV2
                            nc9 = nc9 [OverlappingLeft * 2 : len (nc9)]
                        elif k == 9:
                            nA10 = DatadataV2
                            nc10 = nc10 [OverlappingLeft * 2 : len (nc10)]
                        else:
                            nA11 = DatadataV2
                            nc11 = nc11 [OverlappingLeft * 2 : len (nc11)]
                    else:
                        dataName = "n" + str (k + 2) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeft * 2), OverlappingLeft * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingLeft * 2 - OverlappingLeft)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeft * 100 * 2]
                            counting = counting + 1
                        if k == 11:
                            nA13 = DatadataV2
                            nc13 = nc13 [OverlappingLeft * 2 : len (nc13)]
                        elif k == 12:
                            nA14 = DatadataV2
                            nc14 = nc14 [OverlappingLeft * 2 : len (nc14)]
                        elif k == 13:
                            nA15 = DatadataV2
                            nc15 = nc15 [OverlappingLeft * 2 : len (nc15)]
                        else:
                            nA16 = DatadataV2
                            nc16 = nc16 [OverlappingLeft * 2 : len (nc16)]
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1) 
                        Datadata = eval(dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeft * 2), OverlappingLeft * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingLeft * 2 - OverlappingLeft)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeft * 100 * 2]
                        if k == 0:
                            sdbA1 = DatadataV2
                            sdc1 = sdc1 [OverlappingLeft * 2 : len (sdc1)]
                        elif k == 1:
                            sdbA2 = DatadataV2
                            sdc2 = sdc2 [OverlappingLeft * 2 : len (sdc2)]
                        elif k == 2:
                            sdbA3 = DatadataV2
                            sdc3 = sdc3 [OverlappingLeft * 2 : len (sdc3)]
                        else:
                            sdbA4 = DatadataV2
                            sdc4 = sdc4 [OverlappingLeft * 2 : len (sdc4)]
                        
            """
            # produce the overlapping scenarios for NREM
            """
            if overlapingSideH [a] == 0:
                featuresH = 100 * (OverlappingRightH * 2 + 1)
                for k in range (numberSubjectsN + 1): 
                    if k < 11:
                        dataName = "n" + str (k + 1) 
                        Datadata = eval (dataName) 
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRightH * 2), OverlappingRightH * 2 * 100 + 100))
                        counting = 0 
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRightH * 2 - OverlappingRightH)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRightH * 100 * 2]
                            counting = counting + 1 
                        if k == 0: # subejct 0
                            nH1 = DatadataV2 
                            nch1 = nch1 [0 : len (nch1) - OverlappingRightH * 2] # remove the NREM labels related to the epochs were it is not possible to produce the intended OverlappingRightH window
                        elif k == 1: # subejct 1
                            nH2 = DatadataV2
                            nch2 = nch2 [0 : len (nch2) - OverlappingRightH * 2] 
                        elif k == 2: # subejct 2
                            nH3 = DatadataV2
                            nch3 = nch3 [0 : len (nch3) - OverlappingRightH * 2] 
                        elif k == 3: # subejct 3
                            nH4 = DatadataV2
                            nch4 = nch4 [0 : len (nch4) - OverlappingRightH * 2] 
                        elif k == 4: # subejct 4
                            nH5 = DatadataV2
                            nch5 = nch5 [0 : len (nch5) - OverlappingRightH * 2] 
                        elif k == 5: # subejct 5
                            nH6 = DatadataV2
                            nch6 = nch6 [0 : len (nch6) - OverlappingRightH * 2] 
                        elif k == 6: # subejct 6
                            nH7 = DatadataV2
                            nch7 = nch7 [0 : len (nch7) - OverlappingRightH * 2] 
                        elif k == 7: # subejct 7
                            nH8 = DatadataV2
                            nch8 = nch8 [0 : len (nch8) - OverlappingRightH * 2] 
                        elif k == 8: # subejct 8
                            nH9 = DatadataV2
                            nch9 = nch9 [0 : len (nch9) - OverlappingRightH * 2] 
                        elif k == 9: # subejct 9
                            nH10 = DatadataV2
                            nch10 = nch10 [0 : len (nch10) - OverlappingRightH * 2] 
                        else: # subejct 10
                            nH11 = DatadataV2
                            nch11 = nch11 [0 : len (nch11) - OverlappingRightH * 2] 
                    else:
                        dataName = "n" + str (k + 2)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRightH * 2), OverlappingRightH * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRightH * 2 - OverlappingRightH)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRightH * 100 * 2]
                            counting = counting + 1
                        if k == 11:
                            nH13 = DatadataV2
                            nch13 = nch13 [0 : len (nch13) - OverlappingRightH * 2] 
                        elif k == 12:
                            nH14=DatadataV2
                            nch14 = nch14 [0 : len (nch14) - OverlappingRightH * 2] 
                        elif k == 13:
                            nH15 = DatadataV2
                            nch15 = nch15 [0 : len (nch15) - OverlappingRightH * 2] 
                        else:
                            nH16 = DatadataV2
                            nch16 = nch16 [0 : len (nch16) - OverlappingRightH * 2] 
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1)  
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingRightH * 2), OverlappingRightH * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingRightH * 2 - OverlappingRightH)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingRightH * 100 * 2]
                            counting = counting + 1
                        if k == 0:
                            sdbH1 = DatadataV2
                            sdch1 = sdch1 [0 : len (sdch1) - OverlappingRightH * 2] 
                        elif k == 1:
                            sdbH2 = DatadataV2
                            sdch2 = sdch2 [0 : len (sdch2) - OverlappingRightH * 2] 
                        elif k == 2:
                            sdbH3 = DatadataV2
                            sdch3 = sdch3 [0 : len (sdch3) - OverlappingRightH * 2] 
                        else:
                            sdbH4 = DatadataV2
                            sdch4 = sdch4 [0 : len (sdch4) - OverlappingRightH * 2] 
            elif overlapingSideH [a] == 1:    
                featuresH = 100 * (OverlappingCenterH * 2 + 1)
                for k in range (numberSubjectsN + 1):
                    if k < 11:
                        dataName = "n" + str (k + 1) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenterH * 2), OverlappingCenterH * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenterH, int ((len (Datadata) / 100 - OverlappingCenterH * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenterH * 100 : (x * 100 + 100) + OverlappingCenterH * 100]
                            counting = counting + 1
                        if k == 0:
                            nH1 = DatadataV2
                            nch1 = nch1 [OverlappingCenterH : len (nch1) - OverlappingCenterH]
                        elif k == 1:
                            nH2 = DatadataV2
                            nch2 = nch2 [OverlappingCenterH : len (nch2) - OverlappingCenterH]
                        elif k == 2:
                            nH3 = DatadataV2
                            nch3 = nch3 [OverlappingCenterH : len (nch3) - OverlappingCenterH]
                        elif k == 3:
                            nH4 = DatadataV2
                            nch4 = nch4 [OverlappingCenterH : len (nch4) - OverlappingCenterH]
                        elif k == 4:
                            nH5 = DatadataV2
                            nch5 = nch5 [OverlappingCenterH : len (nch5) - OverlappingCenterH]
                        elif k == 5:
                            nH6 = DatadataV2
                            nch6 = nch6 [OverlappingCenterH : len (nch6) - OverlappingCenterH]
                        elif k == 6:
                            nH7 = DatadataV2
                            nch7 = nch7 [OverlappingCenterH : len (nch7) - OverlappingCenterH]
                        elif k == 7:
                            nH8 = DatadataV2
                            nch8 = nch8 [OverlappingCenterH : len (nch8) - OverlappingCenterH]
                        elif k == 8:
                            nH9 = DatadataV2
                            nch9 = nch9 [OverlappingCenterH : len (nch9) - OverlappingCenterH]
                        elif k == 9:
                            nH10 = DatadataV2
                            nch10 = nch10 [OverlappingCenterH : len (nch10) - OverlappingCenterH]
                        else:
                            nH11 = DatadataV2
                            nch11 = nch11 [OverlappingCenterH : len (nch11) - OverlappingCenterH]
                    else:
                        dataName = "n" + str (k + 2)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenterH * 2), OverlappingCenterH * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenterH, int ((len (Datadata) / 100 - OverlappingCenterH * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenterH * 100 : (x * 100 + 100) + OverlappingCenterH * 100]
                            counting = counting + 1
                        if k == 11:
                            nH13 = DatadataV2
                            nch13 = nch13 [OverlappingCenterH : len (nch13) - OverlappingCenterH]
                        elif k == 12:
                            nH14 = DatadataV2
                            nch14 = nch14 [OverlappingCenterH : len (nch14) - OverlappingCenterH]
                        elif k == 13:
                            nH15 = DatadataV2
                            nch15 = nch15 [OverlappingCenterH : len (nch15) - OverlappingCenterH]
                        else:
                            nH16 = DatadataV2
                            nch16 = nch16 [OverlappingCenterH : len (nch16) - OverlappingCenterH]
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1)
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingCenterH * 2), OverlappingCenterH * 2 * 100 + 100))
                        counting = 0
                        for x in range (OverlappingCenterH, int ((len (Datadata) / 100 - OverlappingCenterH * 2)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) - OverlappingCenterH * 100 : (x * 100 + 100) + OverlappingCenterH * 100]
                            counting = counting + 1
                        if k == 0:
                            sdbH1 = DatadataV2
                            sdch1 = sdch1 [OverlappingCenterH : len (sdch1) - OverlappingCenterH]
                        elif k == 1:
                            sdbH2 = DatadataV2
                            sdch2 = sdch2 [OverlappingCenterH : len (sdch2) - OverlappingCenterH]
                        elif k == 2:
                            sdbH3 = DatadataV2
                            sdch3 = sdch3 [OverlappingCenterH : len (sdch3) - OverlappingCenterH]
                        else:
                            sdbH4 = DatadataV2
                            sdch4 = sdch4 [OverlappingCenterH : len (sdch4) - OverlappingCenterH]
            else: 
                featuresH = 100 * (OverlappingLeftH * 2 + 1)
                for k in range (numberSubjectsN + 1):
                    if k < 11:
                        dataName = "n" + str (k + 1) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeftH * 2), OverlappingLeftH * 2 * 100 + 100))
                        counting = 0
                        for x in range(0, int((len(Datadata)/100-OverlappingLeftH*2-OverlappingLeftH)), 1):
                            DatadataV2 [counting, ]= Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeftH * 100 * 2]
                            counting = counting + 1
                        if k == 0:
                            nH1 = DatadataV2
                            nch1 = nch1 [OverlappingLeftH * 2 : len (nch1)]
                        elif k == 1:
                            nH2 = DatadataV2
                            nch2 = nch2 [OverlappingLeftH * 2 : len (nch2)]
                        elif k == 2:
                            nH3 = DatadataV2
                            nch3 = nch3 [OverlappingLeftH * 2 : len (nch3)]
                        elif k == 3:
                            nH4 = DatadataV2
                            nch4 = nch4 [OverlappingLeftH * 2 : len (nch4)]
                        elif k == 4:
                            nH5 = DatadataV2
                            nch5 = nch5 [OverlappingLeftH * 2 : len (nch5)]
                        elif k == 5:
                            nH6 = DatadataV2
                            nch6 = nch6 [OverlappingLeftH * 2 : len (nch6)]
                        elif k == 6:
                            nH7 = DatadataV2
                            nch7 = nch7 [OverlappingLeftH * 2 : len (nch7)]
                        elif k == 7:
                            nH8 = DatadataV2
                            nch8 = nch8 [OverlappingLeftH * 2 : len (nch8)]
                        elif k == 8:
                            nH9 = DatadataV2
                            nch9 = nch9 [OverlappingLeftH * 2 : len (nch9)]
                        elif k == 9:
                            nH10 = DatadataV2
                            nch10 = nch10 [OverlappingLeftH * 2 : len (nch10)]
                        else:
                            nH11 = DatadataV2
                            nch11 = nch11 [OverlappingLeftH * 2 : len (nch11)]
                    else:
                        dataName = "n" + str (k + 2) 
                        Datadata = eval (dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeftH * 2), OverlappingLeftH * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingLeftH * 2 - OverlappingLeftH)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeftH * 100 * 2]
                            counting = counting + 1
                        if k == 11:
                            nH13 = DatadataV2
                            nch13 = nch13 [OverlappingLeftH * 2 : len (nch13)]
                        elif k == 12:
                            nH14 = DatadataV2
                            nch14 = nch14 [OverlappingLeftH * 2 : len (nch14)]
                        elif k == 13:
                            nH15 = DatadataV2
                            nch15 = nch15 [OverlappingLeftH * 2 : len (nch15)]
                        else:
                            nH16 = DatadataV2
                            nch16 = nch16 [OverlappingLeftH * 2 : len (nch16)]
                if useSDpatients > 0:
                    for k in range (numberSubjectsSD + 1):
                        dataName = "sd" + str (k + 1) 
                        Datadata = eval(dataName)
                        DatadataV2 = np.zeros (((int (len (Datadata) / 100) - OverlappingLeftH * 2), OverlappingLeftH * 2 * 100 + 100))
                        counting = 0
                        for x in range (0, int ((len (Datadata) / 100 - OverlappingLeftH * 2 - OverlappingLeftH)), 1):
                            DatadataV2 [counting, ] = Datadata [(x * 100) : (x * 100 + 100) + OverlappingLeftH * 100 * 2]
                        if k == 0:
                            sdbH1 = DatadataV2
                            sdch1 = sdch1 [OverlappingLeftH * 2 : len (sdch1)]
                        elif k == 1:
                            sdbH2 = DatadataV2
                            sdch2 = sdch2 [OverlappingLeftH * 2 : len (sdch2)]
                        elif k == 2:
                            sdbH3 = DatadataV2
                            sdch3 = sdch3 [OverlappingLeftH * 2 : len (sdch3)]
                        else:
                            sdbH4 = DatadataV2     
                            sdch4 = sdch4 [OverlappingLeftH * 2 : len (sdch4)]
            """
            # create the training and testing sets
            """
            tf.keras.backend.clear_session() # for clearing the secction, realeasing the GPU memory after a training cycle 
            gc.collect() # to release the holded memory in the garbage collector   
            # select the subejcts to compose the training (fist 18 subejct) and testing (last subejct) sets
            if useSDpatients > 0:
                if ee == 0:
                    examinedSubjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0])
                elif ee == 1:
                    examinedSubjects = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1])
                elif ee == 2:
                    examinedSubjects = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2])
                elif ee == 3:
                    examinedSubjects = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 3])
                elif ee == 4:
                    examinedSubjects = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4])
                elif ee == 5:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 5])
                elif ee == 6:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6])
                elif ee == 7:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 7])
                elif ee == 8:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 8])
                elif ee == 9:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 9])
                elif ee == 10:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 10])
                elif ee == 11:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 11])
                elif ee == 12:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 12])
                elif ee == 13:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 13])
                elif ee == 14:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 14])
                elif ee == 15:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 15])
                elif ee == 16:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 16])
                elif ee == 17:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 17])
                else:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            else:
                if ee == 0:
                    examinedSubjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0])
                elif ee == 1:
                    examinedSubjects = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1])
                elif ee == 2:
                    examinedSubjects = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2])
                elif ee == 3:
                    examinedSubjects = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3])
                elif ee == 4:
                    examinedSubjects = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 4])
                elif ee == 5:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5])
                elif ee == 6:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 6])
                elif ee == 7:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7])
                elif ee == 8:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 8])
                elif ee == 9:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 9])
                elif ee == 10:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 10])
                elif ee == 11:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 11])
                elif ee == 12:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 12])
                elif ee == 13:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13])
                else:
                    examinedSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

            XTrain = [] # variable that will hold the A phase train set data
            XTest = [] # variable that will hold the A phase test set data
            XTrainH = [] # variable that will hold the NREM train set data
            XTestH = [] # variable that will hold the NREM test set data
            YTrain = [] # variable that will hold the train set A phase labels
            YTest = [] # variable that will hold the test set A phase labels
            YTrainh = [] # variable that will hold the train set NREM labels
            YTesth = [] # variable that will hold the test set NREM labels
            # initiate the train sets
            if examinedSubjects [Begin] == 0:
                XTrain = nA1
                XTrainH = nH1
                YTrainh = nch1
                YTrain = nc1
            if examinedSubjects [Begin] == 1:
                XTrain = nA2
                XTrainH = nH2
                YTrainh = nch2
                YTrain = nc2
            if examinedSubjects [Begin] == 2:
                XTrain = nA3
                XTrainH = nH3
                YTrainh = nch3
                YTrain = nc3
            if examinedSubjects [Begin] == 3:
                XTrain = nA4
                XTrainH = nH4
                YTrainh = nch4
                YTrain = nc4
            if examinedSubjects [Begin] == 4:
                XTrain = nA5
                XTrainH = nH5
                YTrainh = nch5
                YTrain = nc5
            if examinedSubjects [Begin] == 5:
                XTrain = nA6
                XTrainH = nH6
                YTrainh = nch6
                YTrain = nc6
            if examinedSubjects [Begin] == 6:
                XTrain = nA7
                XTrainH = nH7            
                YTrainh = nch7
                YTrain = nc7
            if examinedSubjects [Begin] == 7:
                XTrain = nA8
                XTrainH = nH8
                YTrainh = nch8
                YTrain = nc8
            if examinedSubjects [Begin] == 8:
                XTrain = nA9
                XTrainH = nH9
                YTrainh = nch9
                YTrain = nc9
            if examinedSubjects [Begin] == 9:
                XTrain = nA10
                XTrainH = nH10
                YTrainh = nch10
                YTrain = nc10
            if examinedSubjects [Begin] == 10:
                XTrain = nA11
                XTrainH = nH11
                YTrainh = nch11
                YTrain = nc11
            if examinedSubjects [Begin] == 11:
                XTrain = nA13
                XTrainH = nH13
                YTrainh = nch13
                YTrain = nc13
            if examinedSubjects [Begin] == 12:
                XTrain = nA14
                XTrainH = nH14
                YTrainh = nch14
                YTrain = nc14
            if examinedSubjects [Begin] == 13:
                XTrain = nA15
                XTrainH = nH15
                YTrainh = nch15
                YTrain = nc15
            if examinedSubjects [Begin] == 14:
                XTrain = nA16
                XTrainH = nH16
                YTrainh = nch16
                YTrain = nc16
            if useSDpatients > 0:
                if examinedSubjects [Begin] == 15:
                    XTrain = sdbA1
                    XTrainH = sdbH1
                    YTrainh = sdch1
                    YTrain = sdc1
                if examinedSubjects [Begin] == 16:
                    XTrain = sdbA2
                    XTrainH = sdbH2
                    YTrainh = sdch2
                    YTrain = sdc2
                if examinedSubjects [Begin] == 17:
                    XTrain = sdbA3
                    XTrainH = sdbH3
                    YTrainh = sdch3
                    YTrain = sdc3
                if examinedSubjects [Begin] == 18:
                    XTrain = sdbA4
                    XTrainH = sdbH4
                    YTrainh = sdch4
                    YTrain = sdc4
            # initiate the test sets    
            if examinedSubjects [BeginTest] == 0:
                XTest = nA1
                XTestH = nH1
                YTesth = nch1
                YTest = nc1
            if examinedSubjects [BeginTest] == 1:
                XTest = nA2
                XTestH = nH2
                YTesth = nch2
                YTest = nc2
            if examinedSubjects [BeginTest] == 2:
                XTest = nA3
                XTestH = nH3
                YTesth = nch3
                YTest = nc3
            if examinedSubjects [BeginTest] == 3:
                XTest = nA4
                XTestH = nH4
                YTesth = nch4
                YTest = nc4
            if examinedSubjects [BeginTest] == 4:
                XTest = nA5
                XTestH = nH5
                YTesth = nch5
                YTest = nc5
            if examinedSubjects [BeginTest] == 5:
                XTest = nA6
                XTestH = nH6
                YTesth = nch6
                YTest = nc6
            if examinedSubjects [BeginTest] == 6:
                XTest = nA7
                XTestH = nH7
                YTesth = nch7
                YTest = nc7
            if examinedSubjects [BeginTest] == 7:
                XTest = nA8
                XTestH = nH8
                YTesth = nch8
                YTest = nc8
            if examinedSubjects [BeginTest] == 8:
                XTest = nA9
                XTestH = nH9
                YTesth = nch9
                YTest = nc9
            if examinedSubjects [BeginTest] == 9:
                XTest = nA10
                XTestH = nH10
                YTesth = nch10
                YTest = nc10
            if examinedSubjects [BeginTest] == 10:
                XTest = nA11
                XTestH = nH11
                YTesth = nch11
                YTest = nc11
            if examinedSubjects [BeginTest] == 11:
                XTest = nA13
                XTestH = nH13
                YTesth = nch13
                YTest = nc13
            if examinedSubjects [BeginTest] == 12:
                XTest = nA14
                XTestH = nH14
                YTesth = nch14
                YTest = nc14
            if examinedSubjects [BeginTest] == 13:
                XTest = nA15
                XTestH = nH15
                YTesth = nch15
                YTest = nc15
            if examinedSubjects [BeginTest] == 14:
                XTest = nA16
                XTestH = nH16
                YTesth = nch16
                YTest = nc16
            if useSDpatients > 0:
                if examinedSubjects [BeginTest] == 15:
                    XTest = sdbA1
                    XTestH = sdbH1
                    YTesth = sdch1
                    YTest = sdc1
                if examinedSubjects [BeginTest] == 16:
                    XTest = sdbA2
                    XTestH = sdbH2
                    YTesth = sdch2
                    YTest = sdc2
                if examinedSubjects [BeginTest] == 17:
                    XTest = sdbA3
                    XTestH = sdbH3
                    YTesth = sdch3
                    YTest = sdc3
                if examinedSubjects [BeginTest] == 18:
                    XTest = sdbA4
                    XTestH = sdbH4
                    YTesth = sdch4
                    YTest = sdc4
            # finish the taining set
            for x in range(20):
                if x < BeginTest and x > Begin : # elements to compose the training set
                    if examinedSubjects [x] == 0:
                        XTrainH = np.concatenate ((XTrainH, nH1), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch1), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA1), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc1), axis = 0)
                    if examinedSubjects [x] == 1:
                        XTrainH = np.concatenate ((XTrainH, nH2), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch2), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA2), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc2), axis = 0)
                    if examinedSubjects [x] == 2:
                        XTrainH = np.concatenate ((XTrainH, nH3), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch3), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA3), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc3), axis = 0)
                    if examinedSubjects [x] == 3:
                        XTrainH = np.concatenate ((XTrainH, nH4), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch4), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA4), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc4), axis = 0)
                    if examinedSubjects [x] == 4:
                        XTrainH = np.concatenate ((XTrainH, nH5), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch5), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA5), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc5), axis = 0)
                    if examinedSubjects [x] == 5:
                        XTrainH = np.concatenate ((XTrainH, nH6), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch6), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA6), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc6), axis = 0)
                    if examinedSubjects [x] == 6:
                        XTrainH = np.concatenate ((XTrainH, nH7), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch7), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA7), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc7), axis = 0)
                    if examinedSubjects [x] == 7:
                        XTrainH = np.concatenate ((XTrainH, nH8), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch8), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA8), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc8), axis = 0)
                    if examinedSubjects [x] == 8:
                        XTrainH = np.concatenate ((XTrainH, nH9), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch9), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA9), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc9), axis = 0)
                    if examinedSubjects [x] == 9:
                        XTrainH = np.concatenate ((XTrainH, nH10), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch10), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA10), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc10), axis = 0)
                    if examinedSubjects [x] == 10:
                        XTrainH = np.concatenate ((XTrainH, nH11), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch11), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA11), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc11), axis = 0)
                    if examinedSubjects [x] == 11:
                        XTrainH = np.concatenate ((XTrainH, nH13), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch13), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA13), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc13), axis = 0)
                    if examinedSubjects [x] == 12:
                        XTrainH = np.concatenate ((XTrainH, nH14), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch14), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA14), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc14), axis = 0)
                    if examinedSubjects [x] == 13:
                        XTrainH = np.concatenate ((XTrainH, nH15), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch15), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA15), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc15), axis = 0)
                    if examinedSubjects [x] == 14:
                        XTrainH = np.concatenate ((XTrainH, nH16), axis = 0)
                        YTrainh = np.concatenate ((YTrainh, nch16), axis = 0)
                        XTrain = np.concatenate ((XTrain, nA16), axis = 0)
                        YTrain = np.concatenate ((YTrain, nc16), axis = 0)
                    if useSDpatients > 0:
                        if examinedSubjects [x] == 15:
                            XTrainH = np.concatenate ((XTrainH, sdbH1), axis = 0)
                            YTrainh = np.concatenate ((YTrainh, sdch1), axis = 0)
                            XTrain = np.concatenate ((XTrain, sdbA1), axis = 0)
                            YTrain = np.concatenate ((YTrain, sdc1), axis = 0)
                        if examinedSubjects [x] == 16:
                            XTrainH = np.concatenate ((XTrainH, sdbH2), axis = 0)
                            YTrainh = np.concatenate ((YTrainh, sdch2), axis = 0)
                            XTrain = np.concatenate ((XTrain, sdbA2), axis = 0)
                            YTrain = np.concatenate ((YTrain, sdc2), axis = 0)
                        if examinedSubjects [x] == 17:
                            XTrainH = np.concatenate ((XTrainH, sdbH3), axis = 0)
                            YTrainh = np.concatenate ((YTrainh, sdch3), axis = 0)
                            XTrain = np.concatenate ((XTrain, sdbA3), axis = 0)
                            YTrain = np.concatenate ((YTrain, sdc3), axis = 0)
                        if examinedSubjects [x] == 18:
                            XTrainH = np.concatenate ((XTrainH, sdbH4), axis = 0)
                            YTrainh = np.concatenate ((YTrainh, sdch4), axis = 0)
                            XTrain = np.concatenate ((XTrain, sdbA4), axis = 0)
                            YTrain = np.concatenate ((YTrain, sdc4), axis = 0)
            del nH1, nch1, nA1, nc1
            del nH2, nch2, nA2, nc2
            del nH3, nch3, nA3, nc3
            del nH4, nch4, nA4, nc4
            del nH5, nch5, nA5, nc5
            del nH6, nch6, nA6, nc6
            del nH7, nch7, nA7, nc7
            del nH8, nch8, nA8, nc8
            del nH9, nch9, nA9, nc9
            del nH10, nch10, nA10, nc10
            del nH11, nch11, nA11, nc11
            del nH13, nch13, nA13, nc13
            del nH14, nch14, nA14, nc14
            del nH15, nch15, nA15, nc15
            del nH16, nch16, nA16, nc16
            if useSDpatients > 0:
                del sdbH1, sdch1, sdbA1, sdc1
                del sdbH2, sdch2, sdbA2, sdc2
                del sdbH3, sdch3, sdbA3, sdc3
                del sdbH4, sdch4, sdbA4, sdc4
            
            # relabel the labels tahat have the three A phase subtypes (A1, A2, and A3, referting to 1, 2, and 3, respectivly) to binary classification
            if APhaseSubtype == 0: # 0 for A and not-A; 1 for A1 and not-A1; 2 for A2 and not-A2; 3 for A3 and not-A3
                for i in range (0, len (YTrain), 1): # check the set with the training A phase labels
                    if YTrain [i] > 0: # identify the occurance an A phase
                        YTrain [i] = 1 # convert the A phase subtypes labels to just A phase labels
                    else:
                        YTrain [i] = 0 # if not-A epoch then use the value 0
                for i in range (0, len (YTest), 1):
                    if YTest [i] > 0:
                        YTest [i] = 1
                    else:
                        YTest [i] = 0
            elif APhaseSubtype == 1: 
                for i in range (0, len (YTrain), 1): 
                    if YTrain [i] > 1:
                        YTrain [i] = 0 
                    elif YTrain [i] == 1:
                        YTrain [i] = 1
                    else:
                        YTrain [i] = 0
                for i in range (0, len (YTest), 1):
                    if YTest [i] > 1:
                        YTest [i] = 0
                    elif YTest [i] == 1:
                        YTest [i] = 1
                    else:
                        YTest [i] = 0
            elif APhaseSubtype == 2: 
                for i in range (0, len (YTrain), 1): 
                    if YTrain [i] > 2:
                        YTrain [i] = 0 
                    elif YTrain [i] == 2:
                        YTrain [i] = 1
                    else:
                        YTrain [i] = 0
                for i in range (0, len (YTest), 1):
                    if YTest [i] > 2:
                        YTest [i] = 0
                    elif YTest [i] == 2:
                        YTest [i] = 1
                    else:
                        YTest [i] = 0
            else:
                for i in range (0, len (YTrain), 1): 
                    if YTrain [i] < 3:
                        YTrain [i] = 0 
                    elif YTrain [i] == 3:
                        YTrain [i] = 1
                    else:
                        YTrain [i] = 0
                for i in range (0, len (YTest), 1):
                    if YTest [i] < 3:
                        YTest [i] = 0
                    elif YTest [i] == 3:
                        YTest [i] = 1
                    else:
                        YTest [i] = 0
            """
            # perform earcly stopping
            """
            class EarlyStoppingAtMinLoss (tf.keras.callbacks.Callback): # class to perform the early stopping
                def __init__ (self, patienteceValue, valid_data): # initialization of the class
                    super (EarlyStoppingAtMinLoss, self).__init__ ()
                    self.patience = patienteceValue # patience value for the early stopping procedure, defining the maximum number of iteration without an increasse of at least "thresholdEarlyStoping" in the AUC before stopping the training procedure
                    self.best_weights = None # best weights of the network
                    self.validation_data = valid_data # data used to validate the model
                def on_train_begin (self, logs = None): # initialize the control parametrers
                    self.wait = 0 # variable holding the number of training iterations without an increasse of at least "thresholdEarlyStoping" in the AUC before stopping the training procedure
                    self.stopped_epoch = 0 # variable hold the value of the training epoch where the model early stoped
                    self.best = 0.2 # initialization of the variable holding the identified best AUC
                    self._data = [] # variable holding the data
                    self.curentAUC = 0.2 # initialization of the variable holding the AUC of the curent training epoch
                    print ('Train started')
                def on_epoch_end (self, epoch, logs = None): # examination at the end of a training epoch
                    X_val, y_val = self.validation_data [0], self.validation_data [1] # load the validation data
                    y_predict = np.asarray (model.predict (X_val)) # variable storing the model's preditions
                    fpr_keras, tpr_keras, thresholds_keras = roc_curve (np.argmax (y_val, axis = 1), y_predict [:, 1]) # produce the receiving operating curve
                    auc_keras = auc (fpr_keras, tpr_keras) # estimate the AUC
                    self.curentAUC = auc_keras # save the current AUC
                    print ('AUC : ', auc_keras)
                    if np.greater(self.curentAUC, self.best+thresholdEarlyStoping): # save the weights if the current AUC is at lest "thresholdEarlyStoping" better than the preivously identifed best AUC
                        print ('Update')
                        self.best = self.curentAUC # update the currently best AUC
                        self.wait = 0 # restart the counting variable for the early stopping procedure
                        self.best_weights = self.model.get_weights () # save the weights of the identified best model
                    else: # the estimated AUC was not at least "thresholdEarlyStoping" better than the preivously identifed best AUC
                        self.wait += 1 # increasse the counting variable for the early stopping procedure
                        if self.wait >= self.patience: # early stop the training if the number of training epochs without a minimum AUC increasse of "thresholdEarlyStoping" was higher than the defined patience value
                            self.stopped_epoch = epoch # save the training epoch were the model early stopped
                            self.model.stop_training = True # flag to identify an early stop
                            print('Restoring model weights from the end of the best epoch')
                            self.model.set_weights (self.best_weights) # restore the weights of the identified best model
                def on_train_end (self, logs = None): # precedure performed at the end of the training
                    if self.stopped_epoch > 0: # report if early stopping occured
                        print ('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
            """
            # A phase classification
            """
            class_weights = class_weight.compute_class_weight ('balanced', np.unique(YTrain), YTrain) # identigy the distribution of the labels to compute the class weights, used for the cost sensitive learning
            class_weights = {i : class_weights [i] for i in range (2)} # reshape the variable to the shape used by the CNN model
            XTrain = XTrain.reshape (len (XTrain), features, 1) # reahspe the data to have "number of epochs" X "number of feature per epoch" X "number of dimentions (chosen to be 1 to form a 1 dimetional array)"
            XTest = XTest.reshape (len (XTest), features, 1)       
            YTrain = to_categorical (YTrain) # convert the labels to categorical
            YTest = to_categorical (YTest)
            x_train, x_valid, y_train, y_valid = train_test_split(XTrain, YTrain, test_size = 0.33, shuffle = True) # produce the validation set
            del XTrain, YTrain # free the memory of currently unnecessary variables
            # classifier: One-Dimensional Convolutional Neural Network (1D-CNN)
            if overlapingSide [a] == 0: # classifier for the A phase with overlapping on the right 
                model = Sequential ()
                model.add (Conv1D (128, KernelSize, strides=1, activation = 'relu', input_shape = (features , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (128*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (100, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            elif overlapingSide [a] == 1: # classifier for the A phase with overlapping on the right and left
                model = Sequential ()
                model.add (Conv1D (64, KernelSize, strides=1, activation = 'relu', input_shape = (features , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (64*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (64*2*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (150, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            else: # classifier for the A phase with overlapping on theleft
                model = Sequential ()
                model.add (Conv1D (32, KernelSize, strides=1, activation = 'relu', input_shape = (features , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (32, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (150, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = [tf.keras.metrics.AUC()]) # compile the classifier
            model.fit(x_train, y_train, batch_size = 1000, epochs = 50, validation_data = (x_valid, y_valid), verbose = 1, shuffle = True, class_weight = class_weights, callbacks = EarlyStoppingAtMinLoss (patienteceValue, (x_valid, y_valid))) # fit the model
            model.save(str (disorder) + '_subtype_' + str(APhaseSubtype) + '_APhase_OverlappingSide_' + str (overlapingSide [a]) + "_subject_" + str(ee) +"_epoch_" + str(ff)) # save the trained model
            print ("\n\n Testing A phase for subject ", ee, ", epoch ", ff, " and overlapping side ", a)    
            proba = model.predict (XTest) # estimate the probability of the labels of the testing set
            YTestOneLine = np.zeros (len (YTest)) # variable to hold the testing set labels in the form of an array
            for x in range (len (YTest)): # convert the categorical labels into an array of fabels to be used by the "confusion_matrix" operation
                if YTest [x, 0] == 1: # column 0 holds the variation of the not-A labels, hence the occurance of 1 is not-A, further denoted as 0 in "YTestOneLine", and the occuranceof 0 is A, further denoted as 1 in "YTestOneLine"
                    YTestOneLine [x] = 0 # not-A label
                else:
                    YTestOneLine [x] = 1 # A label
            predictiony_pred = np.zeros (len (YTestOneLine)) # variable that will hold the A phase labels predicted by the classifier
            for x in range (len (YTestOneLine)): # check all epochs
                if proba [x, 0] > 0.5: # examine the classification threshold
                    predictiony_pred [x] = 0 # predicted not-A
                else:
                    predictiony_pred [x] = 1 # predicted A
            fpr_keras, tpr_keras, thresholds_keras = roc_curve (YTestOneLine, proba[:,1]) # produce the receiving operating curve
            auc_keras = auc (fpr_keras, tpr_keras) # estimate the  AUC
            capPredictedPredicted = predictiony_pred # save the A phase predictions
            """
            # post-processing procedure
            """
            for k in range (len (capPredictedPredicted) - 1): # check the classified epochs to be corrected
                if k > 0: # the procedure cannot be applied to the first and last epoch because it consides the previous and next epoch for the correction
                    if capPredictedPredicted [k - 1] == 0 and capPredictedPredicted [k] == 1 and capPredictedPredicted [k + 1] == 0: # correct 010 to 000
                        capPredictedPredicted [k] = 0 # corrected label
            for k in range (len (capPredictedPredicted) - 1):
                if k > 0:
                    if capPredictedPredicted [k - 1] == 1 and capPredictedPredicted [k] == 0 and capPredictedPredicted [k + 1] == 1: # correct 101 to 111
                        capPredictedPredicted [k] = 1
            # assess the A phase classification performance
            tn, fp, fn, tp = confusion_matrix (YTestOneLine, capPredictedPredicted).ravel() # assess the performance of the classifier
            print (classification_report (YTestOneLine, capPredictedPredicted))
            accuracy0 = (tp + tn) / (tp + tn + fp + fn) # estimate the accuracy
            print ('Accuracy : ', accuracy0)
            sensitivity0 = tp / (tp + fn) # estimate the sensitivity
            print ('Sensitivity : ', sensitivity0)
            specificity0 = tn / (fp + tn) # estimate the specificity
            print ('Specificity : ', specificity0)
            print ('AUC : ', auc_keras)
            # save the cycle's results
            AccAtInterA [ff, a] = accuracy0
            SenAtInterA [ff, a] = sensitivity0
            SpeAtInterA [ff, a] = specificity0
            AUCAtInterA [ff, a] = auc_keras 
            TPInterA [ff, a] = tp
            TNInterA [ff, a] = tn
            FPInterA [ff, a] = fp
            FNInterA [ff, a] = fn        
            PPVInterA [ff, a] = tp / (tp + fp) # positive predictive value
            NPVInterA [ff, a] = tn / (tn + fn) # negative predictive value
            if a == 0: # variable holding the prediction of each CNN composing the classifier ensemble for the A phase estimation
                PredictionYA0 = proba [:, 1] # results for the overlapping on the right
            elif a == 1: 
                PredictionYA1 = proba [:, 1] # results for the overlapping on the left and right
            else:
                PredictionYA2 = proba [:, 1] # results for the overlapping on the left
            del x_train, x_valid, y_train, y_valid # free the memory of currently unnecessary variables
            """
            # NREM classification
            """
            class_weightsh = class_weight.compute_class_weight ('balanced', np.unique(YTrainh), YTrainh)
            class_weightsh = {i : class_weightsh [i] for i in range (2)}
            XTrainH = XTrainH.reshape (len (XTrainH), featuresH, 1)
            XTestH = XTestH.reshape (len (XTestH), featuresH, 1)    
            YTrainh = to_categorical (YTrainh)
            YTesth = to_categorical (YTesth)
            x_trainh, x_validh, y_trainh, y_validh = train_test_split (XTrainH, YTrainh, test_size = 0.33, shuffle = True)
            del XTrainH, YTrainh
            if overlapingSide [a] == 0: # classifier for the NREM with overlapping on the right 
                model = Sequential ()
                model.add (Conv1D (128, KernelSize, strides=1, activation = 'relu', input_shape = (featuresH , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (128, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (150, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            elif overlapingSide [a] == 1: # classifier for the NREM with overlapping on the right and left
                model = Sequential ()
                model.add (Conv1D (32, KernelSize, strides=1, activation = 'relu', input_shape = (featuresH , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (32*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (32*2*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (32*2*2*2, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (100, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            else: # classifier for the NREM with overlapping on theleft
                model = Sequential ()
                model.add (Conv1D (32, KernelSize, strides=1, activation = 'relu', input_shape = (featuresH , 1)))
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Conv1D (32, KernelSize, activation = 'relu')) 
                model.add (MaxPooling1D (pool_size = 2, strides = 2))
                model.add (Dropout (0.1))
                model.add (Flatten ())
                model.add (Dense (150, activation = 'relu'))
                model.add (Dense (2, activation = 'softmax'))
            model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = [tf.keras.metrics.AUC ()])
            model.fit (x_trainh, y_trainh, batch_size=1000, epochs=50, validation_data = (x_validh, y_validh), verbose = 1, shuffle = True, class_weight = class_weightsh, callbacks = EarlyStoppingAtMinLoss (patienteceValue, (x_validh, y_validh)))
            model.save(str (disorder) + '_subtype_' + str(APhaseSubtype) + '_NREM_OverlappingSide_' + str (overlapingSide [a]) + "_subject_" + str(ee) +"_epoch_" + str(ff))
            del x_trainh, x_validh, y_trainh, y_validh  
            print ("\n\n Testing NREM for subject ", ee, ", epoch ", ff, " and overlapping side ", a)    
            proba2 = model.predict (XTestH)
            YTesthOneLineh = np.zeros (len (YTesth));
            for x in range (len (YTesth)):
                if YTesth [x, 0] == 1:
                    YTesthOneLineh [x] = 0
                else:
                    YTesthOneLineh [x] = 1
            predictiony_predh = np.zeros (len (YTesthOneLineh));
            for x in range (len (YTesthOneLineh)):
                if proba2 [x, 0] > 0.5:
                    predictiony_predh [x] = 0
                else:
                    predictiony_predh [x] = 1
            tn, fp, fn, tp = confusion_matrix (YTesthOneLineh, predictiony_predh).ravel()
            fpr_keras, tpr_keras, thresholds_keras = roc_curve (YTesthOneLineh, proba2 [:, 1])
            auc_keras = auc (fpr_keras, tpr_keras)
            capPredictedPredictedh = predictiony_predh
            for k in range (len (capPredictedPredictedh) - 1):
                if k > 0:
                    if capPredictedPredictedh [k - 1] == 0 and capPredictedPredictedh [k] == 1 and capPredictedPredictedh [k + 1] == 0:
                        capPredictedPredictedh [k] = 0
                        
            for k in range (len (capPredictedPredictedh)-1):
                if k > 0:
                    if capPredictedPredictedh [k - 1] == 1 and capPredictedPredictedh [k] == 0 and capPredictedPredictedh [k + 1] == 1:
                        capPredictedPredictedh [k] = 1
            tn, fp, fn, tp = confusion_matrix (YTesthOneLineh, capPredictedPredictedh).ravel()           
            print (classification_report (YTesthOneLineh, capPredictedPredictedh))
            accuracy0 = (tp + tn) / (tp + tn + fp + fn)
            print ('Accuracy : ', accuracy0)
            sensitivity0 = tp / (tp + fn)
            print ('Sensitivity : ', sensitivity0)
            specificity0 = tn / (fp + tn)
            print ('Specificity : ', specificity0)
            print ('AUC : ', auc_keras) 
            AccAtInterN [ff, a] = accuracy0
            SenAtInterN [ff, a] = sensitivity0
            SpeAtInterN [ff, a] = specificity0
            AUCAtInterN [ff, a] = auc_keras       
            TPInterN [ff, a] = tp
            TNInterN [ff, a] = tn
            FPInterN [ff, a] = fp
            FNInterN [ff, a] = fn   
            PPVInterN [ff, a] = tp / (tp + fp)
            NPVInterN [ff, a] = tn / (tn + fn)
            if a == 0:
                PredictionYN0 = proba2 [:, 1]
            elif a == 1: 
                PredictionYN1 = proba2 [:, 1]
            else:
                PredictionYN2 = proba2 [:, 1]
            """
            # Examine the CAP cycles
            """
            examineCAP = 0 # variable to controle when the model can perform the CAP cycle examination
            if len(overlapingSide) > 1: # model with overlapping
                if a > 1: # already checked the tree types of overlapping and now it is going for the majority voting to combine the information
                    """
                    # algiment contant for the A phase and NREM
                    """
                    # align corection for the right
                    if OverlappingRight * 2 >= OverlappingCenter:
                        if OverlappingRightH * 2 >= OverlappingCenterH:
                            if OverlappingRight * 2 >= OverlappingRightH * 2:
                                CorrectRightA = 0
                                CorrectRightH = OverlappingRight * 2 - OverlappingRightH * 2
                            else:
                                CorrectRightA = OverlappingRightH * 2 - OverlappingRight * 2
                                CorrectRightH = 0
                        else:
                            if OverlappingRight * 2 >= OverlappingCenterH * 2:
                                CorrectRightA = 0
                                CorrectRightH = OverlappingRight * 2 - OverlappingCenterH * 2
                            else:
                                CorrectRightA = OverlappingCenterH * 2 - OverlappingRight * 2
                                CorrectRightH = 0
                    else:
                        if OverlappingRightH * 2 >= OverlappingCenterH:
                            if OverlappingCenter * 2 >= OverlappingRightH * 2:
                                CorrectRightA = 0
                                CorrectRightH = OverlappingCenter * 2 - OverlappingRightH * 2
                            else:
                                CorrectRightA = OverlappingRightH * 2 - OverlappingCenter * 2
                                CorrectRightH = 0
                        else:
                            if OverlappingCenter * 2 >= OverlappingCenterH * 2:
                                CorrectRightA = 0
                                CorrectRightH = OverlappingCenter * 2 - OverlappingCenterH * 2
                            else:
                                CorrectRightA = OverlappingCenterH * 2 - OverlappingCenter * 2
                                CorrectRightH = 0
                    # align corection for the left
                    if OverlappingLeft * 2 >= OverlappingCenter:
                        if OverlappingLeftH * 2 >= OverlappingCenterH:
                            if OverlappingLeft * 2 >= OverlappingLeftH * 2:
                                CorrectLeftA = 0
                                CorrectLeftH = OverlappingLeft * 2 - OverlappingLeftH * 2
                            else:
                                CorrectLeftA = OverlappingLeftH * 2 - OverlappingLeft * 2
                                CorrectLeftH = 0
                        else:
                            if OverlappingLeft * 2 >= OverlappingCenterH * 2:
                                CorrectLeftA = 0
                                CorrectLeftH = OverlappingLeft * 2 - OverlappingCenterH * 2
                            else:
                                CorrectLeftA = OverlappingCenterH * 2 - OverlappingLeft * 2
                                CorrectLeftH = 0
                    else:
                        if OverlappingLeftH * 2 >= OverlappingCenterH:
                            if OverlappingCenter * 2 >= OverlappingLeftH * 2:
                                CorrectLeftA = 0
                                CorrectLeftH = OverlappingCenter * 2 - OverlappingLeftH * 2
                            else:
                                CorrectLeftA = OverlappingLeftH * 2 - OverlappingCenter * 2
                                CorrectLeftH = 0
                        else:
                            if OverlappingCenter * 2 >= OverlappingCenterH * 2:
                                CorrectLeftA = 0
                                CorrectLeftH = OverlappingCenter * 2 - OverlappingCenterH * 2
                            else:
                                CorrectLeftA = OverlappingCenterH * 2 - OverlappingCenter * 2
                                CorrectLeftH = 0
                    """
                    # create the array with the classified epochs, if no overlapping was used then two of the prediction arrays will composed of only zeros and the majority voting will copy the results of either the A phase or the NREM assessment
                    """
                    # align the labels
                    if OverlappingRight * 2 >= OverlappingCenter: # identify the largest overlapping to the right
                        OverlappingRightCorrectR = 0
                        OverlappingCenterCorrectR = (OverlappingRight * 2) - OverlappingCenter
                        OverlappingLeftCorrectR = OverlappingRight * 2
                    else:
                        OverlappingRightCorrectR = OverlappingCenter - (OverlappingRight * 2)
                        OverlappingCenterCorrectR = 0
                        OverlappingLeftCorrectR = OverlappingCenter 
                    if OverlappingLeft * 2 >= OverlappingCenter: # identify the largest overlapping to the left
                        OverlappingRightCorrectL = OverlappingLeft * 2
                        OverlappingCenterCorrectL = (OverlappingLeft * 2) - OverlappingCenter
                        OverlappingLeftCorrectL = 0
                    else:
                        OverlappingRightCorrectL = OverlappingCenter * 2
                        OverlappingCenterCorrectL = 0
                        OverlappingLeftCorrectL = (OverlappingCenter * 2) - OverlappingLeft
                    PredictionYA0R = PredictionYA0 [OverlappingRightCorrectL : len (PredictionYA0) - OverlappingRightCorrectR] # overlapping right
                    PredictionYA1R = PredictionYA1 [OverlappingCenterCorrectL : len (PredictionYA1) - OverlappingCenterCorrectR] # overlapping center
                    PredictionYA2R = PredictionYA2 [OverlappingLeftCorrectL : len (PredictionYA2) - OverlappingLeftCorrectR] # overlapping left
                    PredictionYA3 = np.asarray ([PredictionYA0R, PredictionYA1R, PredictionYA2R]) # combine the three arrays to form a matrix
                    PredictionYA3 = PredictionYA3[: , CorrectRightA : len (PredictionYA2R) - CorrectLeftA] # align the A phase and NREM predictions
                    # Weighted voting
                    cappredictiony_predhA = np.zeros (len (PredictionYA3 [0])) # arry that wil contain the majority vonting output
                    for combinationA in range (0, len (cappredictiony_predhA), 1): # check line by line the output of the classifiers
                        # cappredictiony_predhATemp = np.sum (PredictionYA3 [:, combinationA] * AUCAtInterA) # variable to hold the sum of all predictions of each epoch  for the majority voting
                        cappredictiony_predhATemp = np.sum (PredictionYA3 [:, combinationA])
                        if cappredictiony_predhATemp >= 1.5: # majority voting, if two of the three classifiers predicted 1 then it is 1 otherwise leave the 0
                            cappredictiony_predhA [combinationA] = 1
                    YTestOneLine = YTestOneLine [OverlappingLeftCorrectL : len (YTestOneLine) - OverlappingLeftCorrectR] # alingh the labels, takinh into consideration that they are currently alingn with the overlapping to the left scenario (a = 2)
                    YTestOneLine = YTestOneLine[CorrectRightA : len (YTestOneLine) - CorrectLeftA] # align the A phase and NREM predictions
                    if useSDpatients > 0:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Right_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) # save the outputs for further examination
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA0R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA1R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Left_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA2R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA3, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_CombinationStrategy_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhA, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseAPhase_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (YTestOneLine, f)
                        f.close ()
                    else:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Right_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize)  # save the outputs for further examination
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA0R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Center_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA1R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Left_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA2R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYA3, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_CombinationStrategy_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhA, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseAPhase_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (YTestOneLine, f)
                        f.close ()
                    # for NREM   
                    if OverlappingRightH * 2 >= OverlappingCenterH: # identify the largest overlapping to the right
                        OverlappingRightCorrectR = 0
                        OverlappingCenterCorrectR = (OverlappingRightH * 2) - OverlappingCenterH
                        OverlappingLeftCorrectR = OverlappingRightH * 2
                    else:
                        OverlappingRightCorrectR = OverlappingCenterH - (OverlappingRightH * 2)
                        OverlappingCenterCorrectR = 0
                        OverlappingLeftCorrectR = OverlappingCenterH
                    if OverlappingLeftH * 2 >= OverlappingCenterH: # identify the largest overlapping to the left
                        OverlappingRightCorrectL = OverlappingLeftH * 2
                        OverlappingCenterCorrectL = (OverlappingLeftH * 2) - OverlappingCenterH
                        OverlappingLeftCorrectL = 0
                    else:
                        OverlappingRightCorrectL = OverlappingCenterH * 2
                        OverlappingCenterCorrectL = 0
                        OverlappingLeftCorrectL = (OverlappingCenterH * 2) - OverlappingLeftH
                    PredictionYN0R = PredictionYN0 [OverlappingRightCorrectL : len (PredictionYN0) - OverlappingRightCorrectR]
                    PredictionYN1R = PredictionYN1 [OverlappingCenterCorrectL : len (PredictionYN1) - OverlappingCenterCorrectR]
                    PredictionYN2R = PredictionYN2 [OverlappingLeftCorrectL : len (PredictionYN2) - OverlappingLeftCorrectR]
                    PredictionYN3 = np.asarray ([PredictionYN0R, PredictionYN1R, PredictionYN2R])
                    PredictionYN3 = PredictionYN3[: , CorrectRightH : len (PredictionYN2R) - CorrectLeftH]
                    cappredictiony_predhN = np.zeros (len (PredictionYN3 [0]))
                    for combinationN in range (0, len (cappredictiony_predhN), 1): 
                        # cappredictiony_predhNTemp = np.sum (PredictionYN3 [:, combinationN] * AUCAtInterN) 
                        cappredictiony_predhNTemp = np.sum (PredictionYN3 [:, combinationN]) 
                        if cappredictiony_predhNTemp >= 1.5: 
                            cappredictiony_predhN [combinationN] = 1
                    YTesthOneLineh = YTesthOneLineh [OverlappingLeftCorrectL : len (YTesthOneLineh) - OverlappingLeftCorrectR]
                    YTesthOneLineh = YTesthOneLineh[CorrectRightH : len (YTesthOneLineh) - CorrectLeftH] # align the A phase and NREM predictions
                    if useSDpatients > 0:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Right_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize)  # save the outputs for further examination
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN0R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Center_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN1R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Left_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN2R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN3, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_CombinationStrategy_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhN, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseNREM_Combined_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (YTesthOneLineh, f)
                        f.close ()
                    else:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Right_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize)  # save the outputs for further examination
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN0R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Center_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN1R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Left_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN2R, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (PredictionYN3, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_CombinationStrategy_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhN, f)
                        f.close ()
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseNREM_Combined_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (YTesthOneLineh, f)
                        f.close ()
                    examineCAP = 1 # perform the CAP examination
                    """
                    # performance of the essemble for the A phase estimation
                    """
                    for k in range (len(cappredictiony_predhN)): # decrese the missclassification by converting the classified A phases into not-A when the NREM classifiers indicates a period of REM/wake (A phase can only occur during NREM sleep)
                        if cappredictiony_predhN[k]==0:
                            cappredictiony_predhA[k]=0
                    for k in range (len (cappredictiony_predhA) - 1):
                        if k > 0:
                            if cappredictiony_predhA [k - 1] == 0 and cappredictiony_predhA [k] == 1 and cappredictiony_predhA [k + 1] == 0:
                                cappredictiony_predhA [k] = 0
                                
                    for k in range (len (cappredictiony_predhA)-1):
                        if k > 0:
                            if cappredictiony_predhA [k - 1] == 1 and cappredictiony_predhA [k] == 0 and cappredictiony_predhA [k + 1] == 1:
                                cappredictiony_predhA [k] = 1  
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
                    if useSDpatients > 0:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhA, f)
                        f.close ()
                    else:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_PostProcessing_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (cappredictiony_predhA, f)
                        f.close ()
                    """
                    # performance of the essemble for the NREM estimation
                    """
                    for k in range (len (cappredictiony_predhN) - 1):
                        if k > 0:
                            if cappredictiony_predhN [k - 1] == 0 and cappredictiony_predhN [k] == 1 and cappredictiony_predhN [k + 1] == 0:
                                cappredictiony_predhN [k] = 0
                                
                    for k in range (len (cappredictiony_predhN)-1):
                        if k > 0:
                            if cappredictiony_predhN [k - 1] == 1 and cappredictiony_predhN [k] == 0 and cappredictiony_predhN [k + 1] == 1:
                                cappredictiony_predhN [k] = 1
                    # correction procedure for the NREM that will convert all 1 second epochs that are inside a duration lower than 30 s will be converted from 0 (REM/wake) to 1 (NREM)
                    searchval = 1
                    NREMPredicted = np.copy(cappredictiony_predhN) # variable to hold the FSM results
                    # ii = np.where (cappredictiony_predhN == searchval)[0] # find the index of all NREM epochs
                    # jj = [t - s for s, t in zip (ii, ii [1 :])] # find the duration all NREM
                    # counting = 0 # variable to control the minimum number of variations between 1 and 0 inside the standard 30 seconds of data
                    # for i in range (len (ii)): # examine all identified A phases
                    #     if i < len (ii) - 1: # examine untli the last identified A phase
                    #         if jj [i] <= 30: # if the difference is not 1 then it finished the A phase and this new idenfified A phase needs to be less than 60 s apart from the previous to be a valid CAP cycle
                    #             NREMPredicted [ii [i] : ii [i + 1]] = 2 # score the B phase, that is between the A phases as 2 to be aprt of the CAP cycle
                    #             counting = counting + 1 # increasse the number of candidates for valid CAP cycle identified
                    #         elif counting >= 2: # if the difference between teo A phases is longer than 60 s then it finiched the CAP sequence, and if at least two CAP cycles were idenfied as candidate for valid CAP cycle then a correct CAP cycle was found
                    #             NREMPredicted [NREMPredicted == 2] = 1 # convert the scores 2 to 1 to indicate that a valid CAP cycle was found
                    #             counting = 0 # restart the conting for the valid CAP cycles
                    #         else: # the conditions for scoring a valid CAP cycle were not met do clear the scored values
                    #             NREMPredicted [NREMPredicted == 2] = 0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                    #             counting = 0 # restart the conting for the valid CAP cycles
                    #     else: # finished checking the data for the A phases
                    #         NREMPredicted [NREMPredicted == 2] =0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                    # NREMPredicted [NREMPredicted == 2] = 0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                    ("\n\n Testing NREM after ensemble for subject ", ee, ", epoch ", ff)    
                    tn, fp, fn, tp = confusion_matrix (YTesthOneLineh, NREMPredicted).ravel()
                    print (classification_report (YTesthOneLineh, NREMPredicted))
                    accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                    print ('Accuracy : ', accuracy0)
                    sensitivity0 = tp / (tp + fn)
                    print ('Sensitivity : ', sensitivity0)
                    specificity0 = tn / (fp + tn)
                    print ('Specificity : ', specificity0)
                    AccAtInterNC [ff] = accuracy0
                    SenAtInterNC [ff] = sensitivity0
                    SpeAtInterNC [ff] = specificity0    
                    TPInterNC [ff] = tp
                    TNInterNC [ff] = tn
                    FPInterNC [ff] = fp
                    FNInterNC [ff] = fn   
                    PPVInterNC [ff] = tp / (tp + fp)
                    NPVInterNC [ff] = tn / (tn + fn)
                    NREMtotalInter [ff] = sum (NREMPredicted) # total number of epochs classified as NREM
                    if useSDpatients > 0:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_PostProcessing_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (NREMPredicted, f)
                        f.close ()
                    else:
                        StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_PostProcessing_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                        f = open (StringText, 'ab')
                        pickle.dump (NREMPredicted, f)
                        f.close ()
            else: #model without overlapping
                cappredictiony_predhA = PredictionYA1 # variable holding the classified A phase epochs
                cappredictiony_predhN = PredictionYN1 # variable holding the classified NREM epochs
                examineCAP = 1 # perform the CAP examination
                NREMtotalInter [ff] = sum (cappredictiony_predhN) # total number of epochs classified as NREM
                AtotalInter [ff] = sum (capPredictedPredicted) # total number of epochs classified as A phase
                if useSDpatients > 0:
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (cappredictiony_predhA, f)
                    f.close ()
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (cappredictiony_predhN, f)
                    f.close ()
                else:
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedAPhase_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (cappredictiony_predhA, f)
                    f.close ()
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedNREM_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (cappredictiony_predhN, f)
                    f.close ()
            """
            # classify the CAP cycles with the Finite State Machine (FSM)
            """
            if examineCAP == 1:
                # CAP cycles predicted from the estimated A phase
                searchval = 1 # look for A phase, corresponding to 1, while not-A is 0
                predictedCAP = np.copy(cappredictiony_predhA) # variable to be examined by the FSM, holding the classified A phase locations                             
                capPredicted = np.zeros (len (predictedCAP)) # variable to hold the FSM results
                ii = np.where (predictedCAP == searchval)[0] # find the index of all A phases
                jj = [t - s for s, t in zip (ii, ii [1 :])] # find the duration all A phases
                counting = 0 # variable to control the maximum duration of the identified A phase is lower than 60 s
                for i in range (len (ii)): # examine all identified A phases
                    if i < len (ii) - 1: # examine untli the last identified A phase
                        if jj [i] == 1: # identified A phase, 1 indicated that the diference from the previous A phase epoch is 1 s so it is still the same A phase
                            capPredicted [ii [i]] = 4 # score as 4 to mark as a candidate for valid A phase of a CAP cycle
                        elif jj [i] <= 60: # if the difference is not 1 then it finished the A phase and this new idenfified A phase needs to be less than 60 s apart from the previous to be a valid CAP cycle
                            capPredicted [capPredicted == 4] = 2 # score as 2 to mark as a candidate for valid CAP cycle
                            capPredicted [ii [i] : ii [i + 1]] = 2 # score the B phase, that is between the A phases as 2 to be aprt of the CAP cycle
                            counting = counting + 1 # increasse the number of candidates for valid CAP cycle identified
                        elif counting >= 2: # if the difference between teo A phases is longer than 60 s then it finiched the CAP sequence, and if at least two CAP cycles were idenfied as candidate for valid CAP cycle then a correct CAP cycle was found
                            capPredicted [capPredicted == 2] = 1 # convert the scores 2 to 1 to indicate that a valid CAP cycle was found
                            capPredicted [capPredicted == 4] = 0 # convert the scores 4 to 0 to eliminate the A phases that were not scored as part of the valid CAP cycles
                            counting = 0 # restart the conting for the valid CAP cycles
                        else: # the conditions for scoring a valid CAP cycle were not met do clear the scored values
                            capPredicted [capPredicted == 4] = 0 # eliminate the scored A phases that were not part of a valid CAP cycle
                            capPredicted [capPredicted == 2] = 0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                            counting = 0 # restart the conting for the valid CAP cycles
                    else: # finished checking the data for the A phases
                        capPredicted [capPredicted == 4] =0 # eliminate the scored A phases that were not part of a valid CAP cycle
                        capPredicted [capPredicted == 2] =0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                capPredicted [capPredicted == 2] = 0 # eliminate the CAP cycle that was not valid because a minimum sequence of 2 CAP cycles are required to score the CAP cycles as valid 
                # CAP cycles predicted from the database
                searchval = 1
                predictedCAP = YTestOneLine # variable to be examined by the FSM, holding the A phase locations identified on the databse
                capPredictedDatabase = np.zeros (len (predictedCAP))
                ii = np.where (predictedCAP == searchval)[0] 
                jj = [t - s for s, t in zip (ii, ii [1 :])]
                counting = 0
                for i in range (len (ii)):
                    if i < len (ii) - 1:
                        if jj [i] == 1:
                            capPredictedDatabase [ii [i]] = 4 
                        elif jj [i] <= 60: 
                            capPredictedDatabase [capPredictedDatabase == 4] = 2
                            capPredictedDatabase [ii [i] : ii [i + 1]] = 2 
                            counting = counting + 1
                        elif counting >= 2: 
                            capPredictedDatabase [capPredictedDatabase == 2] = 1
                            capPredictedDatabase [capPredictedDatabase == 4] = 0
                            counting = 0
                        else: 
                            capPredictedDatabase [capPredictedDatabase == 4] = 0
                            capPredictedDatabase [capPredictedDatabase == 2] = 0
                            counting = 0
                    else:
                        capPredictedDatabase [capPredictedDatabase == 4] = 0
                        capPredictedDatabase [capPredictedDatabase == 2] = 0
                capPredictedDatabase [capPredictedDatabase == 2] = 0
                # examine the results for the CAP cycle estimation           
                print ("\n\n Testing CAP for subject ", ee, ", epoch ", ff)    
                tn, fp, fn, tp = confusion_matrix (capPredictedDatabase, capPredicted).ravel() # create the confusion matrix of the true negatives, tn, false positives, fp, false negatives, fn, and true positives, tp
                print (' CAP : ')
                print (classification_report (capPredictedDatabase, capPredicted)) # present the confusion matrix report
                accuracy0 = (tp + tn) / (tp + tn + fp + fn) # estimate the accuracy
                print ('Accuracy : ', accuracy0)
                sensitivity0 = tp / (tp + fn) # estimate the sensitivity
                print ('Sensitivity : ', sensitivity0)
                specificity0 = tn / (fp + tn) # estimate the specificity
                print ('Specificity : ', specificity0)
                # save the CAP cycle assessment results of the cycle           
                AccAtInter [ff] = accuracy0
                SenAtInter [ff] = sensitivity0
                SpeAtInter [ff] = specificity0 
                TPInter [ff] = tp
                TNInter [ff] = tn
                FPInter [ff] = fp
                FNInter [ff] = fn 
                PPVInter [ff] = tp / (tp + fp) # positive predictive value
                NPVInter [ff] = tn / (tn + fn) # negative predictive value
                # save the CAP assessment results of the cycle 
                CAPtotalInter [ff] = sum (capPredicted) # total number of epochs classified as part of a CAP cycle
                CAPrateErrototalInter [ff] = (sum (capPredicted) / sum (NREMPredicted)) - (sum (capPredictedDatabase) / sum (YTesthOneLineh)) # estimate the CAP rate error from the difference between the predicted and the true CAP rate
                CAPrateErroPercentagetotalInter [ff] = (abs ((sum (capPredicted) / sum (NREMPredicted)) - (sum (capPredictedDatabase) / sum (YTesthOneLineh)))) / (sum (capPredictedDatabase) / sum (YTesthOneLineh)) # estimate the CAP rate error in percentage
                print('CAP rate error : ', CAPrateErrototalInter [ff])
                print('CAP rate error percentage : ', CAPrateErroPercentagetotalInter [ff])
                if useSDpatients > 0:
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedCAP_FSM_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (capPredicted, f)
                    f.close ()
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseCAP_FSM_subject{}_Epoch{}_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (capPredictedDatabase, f)
                    f.close ()
                else:
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_EstimatedCAP_FSM_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (capPredicted, f)
                    f.close ()
                    StringText = str (disorder) + "_subtype_" + str(APhaseSubtype)+"_DatabaseCAP_FSM_subject{}_Epoch{}-notSDB_KernelSize{}.txt".format(ee, ff, KernelSize) 
                    f = open (StringText, 'ab')
                    pickle.dump (capPredictedDatabase, f)
                    f.close ()
            del XTest, YTest, YTesth # clear the variable hold the data from the previous iteration
            del model
        # save the results for the subejct analysis    
        AccAtEndA [ee,:] = np.mean (AccAtInterA, axis = 0)
        SenAtEndA [ee,:] = np.mean (SenAtInterA, axis = 0)
        SpeAtEndA [ee,:] = np.mean (SpeAtInterA, axis = 0)
        AUCAtEndA [ee,:] = np.mean (AUCAtInterA, axis = 0)
        TPEndA [ee,:] = np.mean (TPInterA, axis = 0)
        TNEndA [ee,:] = np.mean (TNInterA, axis = 0)
        FPEndA [ee,:] = np.mean (FPInterA, axis = 0)
        FNEndA [ee,:] = np.mean (FNInterA, axis = 0)
        PPVEndA [ee,:] = np.mean (PPVInterA, axis = 0)
        NPVEndA [ee,:] = np.mean (NPVInterA, axis = 0)
        AccAtEndN [ee,:] = np.mean (AccAtInterN, axis = 0)
        SenAtEndN [ee,:] = np.mean (SenAtInterN, axis = 0)
        SpeAtEndN [ee,:] = np.mean (SpeAtInterN, axis = 0)
        AUCAtEndN [ee,:] = np.mean (AUCAtInterN, axis = 0)
        TPEndN [ee,:] = np.mean (TPInterN, axis = 0)
        TNEndN [ee,:] = np.mean (TNInterN, axis = 0)
        FPEndN [ee,:] = np.mean (FPInterN, axis = 0)
        FNEndN [ee,:] = np.mean (FNInterN, axis = 0)
        PPVEndN [ee,:] = np.mean (PPVInterN, axis = 0)
        NPVEndN [ee,:] = np.mean (NPVInterN, axis = 0)
        AccAtEndAC [ee] = np.mean (AccAtInterAC)
        SenAtEndAC [ee] = np.mean (SenAtInterAC)
        SpeAtEndAC [ee] = np.mean (SpeAtInterAC)
        TPEndAC [ee] = np.mean (TPInterAC)
        TNEndAC [ee] = np.mean (TNInterAC)
        FPEndAC [ee] = np.mean (FPInterAC)
        FNEndAC [ee] = np.mean (FNInterAC)
        PPVEndAC [ee] = np.mean (PPVInterAC)
        NPVEndAC [ee] = np.mean (NPVInterAC)    
        AccAtEndNC [ee] = np.mean (AccAtInterNC)
        SenAtEndNC [ee] = np.mean (SenAtInterNC)
        SpeAtEndNC [ee] = np.mean (SpeAtInterNC)
        TPEndNC [ee] = np.mean (TPInterNC)
        TNEndNC [ee] = np.mean (TNInterNC)
        FPEndNC [ee] = np.mean (FPInterNC)
        FNEndNC [ee] = np.mean (FNInterNC)
        PPVEndNC [ee] = np.mean (PPVInterNC)
        NPVEndNC [ee] = np.mean (NPVInterNC)
        AccAtEnd [ee] = np.mean (AccAtInter)
        SenAtEnd [ee] = np.mean (SenAtInter)
        SpeAtEnd [ee] = np.mean (SpeAtInter)
        TPEnd [ee] = np.mean (TPInter)
        TNEnd [ee] = np.mean (TNInter)
        FPEnd [ee] = np.mean (FPInter)
        FNEnd [ee] = np.mean (FNInter)
        PPVEnd [ee] = np.mean (PPVInter)
        NPVEnd [ee] = np.mean (NPVInter)
        NREMtotalEnd [ee] = np.mean (NREMtotalInter)
        AtotalEnd [ee] = np.mean (AtotalInter)
        CAPtotalEnd [ee] = np.mean (CAPtotalInter)
        CAPrateErrototalEnd [ee] = np.mean (CAPrateErrototalInter)
        CAPrateErroPercentagetotalEnd [ee] = np.mean (CAPrateErroPercentagetotalInter)
        metricsA = np.c_ [AccAtEndA, SenAtEndA, SpeAtEndA, AUCAtEndA, TPEndA, TNEndA, FPEndA, FNEndA, PPVEndA, NPVEndA] # variable holding all the results for the A phase analysis
        metricsN = np.c_ [AccAtEndN, SenAtEndN, SpeAtEndN, AUCAtEndN, TPEndN, TNEndN, FPEndN, FNEndN, PPVEndN, NPVEndN] # variable holding all the results for the NREM analysis
        metricsAC = np.c_ [AccAtEndAC, SenAtEndAC, SpeAtEndAC, TPEndAC, TNEndAC, FPEndAC, FNEndAC, PPVEndAC, NPVEndAC, AtotalEnd] # variable holding all the results for the A phase analysis after ensemble
        metricsNC = np.c_ [AccAtEndNC, SenAtEndNC, SpeAtEndNC, TPEndNC, TNEndNC, FPEndNC, FNEndNC, PPVEndNC, NPVEndNC, NREMtotalEnd] # variable holding all the results for the NREM analysis after ensemble
        metricsC = np.c_ [AccAtEnd, SenAtEnd, SpeAtEnd, TPEnd, TNEnd, FPEnd, FNEnd, PPVEnd, NPVEnd, CAPtotalEnd, CAPrateErrototalEnd, CAPrateErroPercentagetotalEnd] # variable holding all the results for the CAP cycle and CAP rate analysis
        if useSDpatients > 0:
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsAphaseV2.txt", 'ab') # open (or create if it does not exist) the pikle file in the adding entry mode
            pickle.dump (metricsA, f) # add a new entry to the pickle file
            f.close () # close the pickle file
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsAphaseEnsembleV2.txt", 'ab')
            pickle.dump (metricsAC, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsNREMV2.txt", 'ab')
            pickle.dump (metricsN, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsNREMEnsembleV2.txt", 'ab')
            pickle.dump (metricsNC, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsCAPV2.txt", 'ab')
            pickle.dump (metricsC, f)
            f.close ()
        else:
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsAphaseV2-notSDB.txt", 'ab') # open (or create if it does not exist) the pikle file in the adding entry mode
            pickle.dump (metricsA, f) # add a new entry to the pickle file
            f.close () # close the pickle file
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsAphaseEnsembleV2-notSDB.txt", 'ab')
            pickle.dump (metricsAC, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsNREMV2-notSDB.txt", 'ab')
            pickle.dump (metricsN, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsNREMEnsembleV2-notSDB.txt", 'ab')
            pickle.dump (metricsNC, f)
            f.close ()
            f = open (str (disorder) + "_subtype_" + str(APhaseSubtype)+"_metricsCAPV2-notSDB.txt", 'ab')
            pickle.dump (metricsC, f)
            f.close ()        
print(' Finished ')