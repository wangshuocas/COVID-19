#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import configparser
from sklearn import metrics
from tensorflow.keras.utils import plot_model

import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model

import tensorflow
#sys.path.append("..")
from COVIDGenerator import LungNCPGenerator

MINIMUM_TF_VERSION = 1, 14, 0

custom_objects = {}
        
        
def tf_version():
    """ Get the Tensorflow version.
        Returns
            tuple of (major, minor, patch).
    """
    return tuple(map(int, tf.version.VERSION.split('-')[0].split('.')))


def tf_version_ok(minimum_tf_version=MINIMUM_TF_VERSION):
    """ Check if the current Tensorflow version is higher than the minimum version.
    """
    return tf_version() >= minimum_tf_version
        
def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main(args=None):
    print('test')
    if args is None:
        configPath = sys.argv[1]
    configFile = read_config_file(configPath)
    
    multi_gpuList = json.loads(configFile.get('MODEL', 'multi_gpuList'))
    if len(multi_gpuList) > 1:
        print('only support single GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in multi_gpuList])
    
    modelPath = configFile.get('MODEL', 'weightPath')
    print('modelPath', modelPath)
    
    batchSize = configFile.getint('MODEL', 'batchSize')
    
    pl = configFile.get('MODEL', 'phaseName')
    phaseList = pl.split(',')
    dataDirList = []
    for idx, phase in enumerate(phaseList):
        dataDir1 = configFile.get('MODEL', 'dataDir'+str(idx))
        dataDirList.append(dataDir1)
    
    
    saveProbFlag = configFile.getboolean('MODEL', 'saveProb')
    
    generatorList = []
    for idx, dataDir1 in enumerate(dataDirList):
        generator1 = LungNCPGenerator(
            data_dir = dataDir1,
            #group_method = 'random',
            group_method = 'none',
            shuffle_groups = False, 
            batch_size = batchSize,
            transform_generator = None,
            config=configFile
        )
        generatorList.append(generator1)
        
    runModel = keras.models.load_model(modelPath, custom_objects=custom_objects)
    if saveProbFlag == True:
        runModel = Model(runModel.inputs, [runModel.outputs[0], runModel.get_layer('global_average_pooling3d').output])
    
    truthAllD, predAllD, IDAllD, featureAllD = [], [], [], []
    for pidx, phase in enumerate(phaseList):
        truththisD, predthisD, IDthisD, featurethisD = [], [], [], []
        generator1 = generatorList[pidx]
        multi_enqueuerTest = tensorflow.keras.utils.OrderedEnqueuer(generator1, use_multiprocessing=configFile.getboolean('GENERATOR', 'multiprocessing'))
        multi_enqueuerTest.start(workers=configFile.getint('GENERATOR', 'workers'), 
                         max_queue_size=configFile.getint('GENERATOR', 'max_queue_size'))
        gTe = multi_enqueuerTest.get()
        
        for batchi in range(generator1.__len__()):
            inputs, targetsOri = next(gTe)
            if saveProbFlag == True:
                targets, IDthis = targetsOri
            else:
                targets = targetsOri
            truthLabel = targets
            thisPred = runModel.predict_on_batch(inputs)
            if saveProbFlag == True:
                NPCPredBatchGlobal, feature = thisPred
                #print('feature', feature.shape, IDthis, type(IDthis))
                IDthisD.append(IDthis)
                featurethisD.append(feature)
                NPCPredBatchGlobal = NPCPredBatchGlobal[:,0]
            else:
                NPCPredBatchGlobal = thisPred
                NPCPredBatchGlobal = NPCPredBatchGlobal.numpy()[:,0]
           
            truththisD.append(truthLabel)
            predthisD.append(NPCPredBatchGlobal)
        multi_enqueuerTest.stop()
        truththisD = np.concatenate(truththisD, axis=0)
        predthisD = np.concatenate(predthisD, axis=0)
        truthAllD.append(truththisD)
        predAllD.append(predthisD)
        
        if saveProbFlag == True:
            IDthisD = np.concatenate(IDthisD, axis=0)
            featurethisD = np.concatenate(featurethisD, axis=0)
            IDAllD.append(IDthisD)
            featureAllD.append(featurethisD)
        testAuc, testAcc, testSen, testSpe = _calAuc(truththisD, predthisD)
        print(phase, 'auc, acc, sen, spe', testAuc, testAcc, testSen, testSpe)
        
    if saveProbFlag == True:        
        fsaveProb = open('prob_predicted.csv', 'w')
        fsaveProb.write('phase,ID,pred,label,' + ','.join(['feature'+str(i) for i in range(featureAllD[0].shape[1])]) + '\n')
        for pidx in range(len(IDAllD)):
            phase = phaseList[pidx]
            featurethisD = featureAllD[pidx]
            IDthisD = IDAllD[pidx]
            predthisD = predAllD[pidx]
            truththisD = truthAllD[pidx]
            for ii in range(IDthisD.shape[0]):
                fsaveProb.write(phase+',%s,%.6f,%d' % (IDthisD[ii], predthisD[ii], truththisD[ii]))
                for fi in range(featurethisD.shape[1]):
                    fsaveProb.write(',%.6f' % (featurethisD[ii,fi]))
                fsaveProb.write('\n')
        fsaveProb.close()

def _calAuc(trueList, predList, co=0.8):
    predList = predList.squeeze()
    trueList = trueList.squeeze().astype(np.uint8)
    fpr, tpr, thresholds = metrics.roc_curve(trueList, predList, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    predListC = np.zeros_like(trueList)
    predListC[predList > co] = 1#
    print('tp', trueList.shape, trueList.min(), trueList.max(), predListC.shape, predListC.min(), predListC.max())
    acc = metrics.accuracy_score(trueList, predListC) * 100
    tn, fp, fn, tp = metrics.confusion_matrix(trueList, predListC).ravel()
    testSen = tp/float(tp+fn)
    testSpe = tn/float(tn+fp)
    return auc, acc, testSen, testSpe
    
    
if __name__ == '__main__':
    ###############use this command line to test##########
    #python test.py config_test.ini
    ######################################################
    main()
        