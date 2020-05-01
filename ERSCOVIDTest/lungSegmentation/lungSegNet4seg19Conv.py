# -*- coding: utf-8 -*-

import os
#import cPickle
import sys
import scipy
import numpy as np
import scipy.io as sio
import SimpleITK as sitk

from scipy.ndimage.interpolation import zoom
from skimage.segmentation import find_boundaries

#please install segmentation-models  1.0.1
#https://github.com/qubvel/segmentation_models
import segmentation_models as sm

            
def _contour(mask):
    target = find_boundaries(mask, mode='inner')
    maskedTarget = np.ma.masked_where(target == 0, target)
    return maskedTarget
            
class SegLung():
    def __init__(self):
        pass
        
    def pred(self, argvList):
        argDic = {}
        for ar in argvList:
            k,v = ar.split('=')
            argDic[k] = v
        gpu = argDic['gpu']
        dataset = argDic['dataset']
        multiprocess = argDic['multiprocess']
        processID, processNum = multiprocess.split('/')
        processID = int(processID)
        processNum = int(processNum)
        assert processID < processNum
        
        rootCTDir = dataset
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu
        
        TARSIZE = 256.0
        weights_path = '../model/lungSegmentation.hdf5'
        BACKBONE = 'densenet121'
        model = sm.FPN(BACKBONE, classes=1, activation='sigmoid', input_shape=(256, 256, 3), encoder_weights='imagenet')
        model.load_weights(weights_path, by_name=False)
        cnt = 0
        
        
        IDList = os.listdir(rootCTDir)
        IDList = sorted(IDList)
        seglen = len(IDList)//processNum
        processfifo = []
        for mi in range(processNum):
            processfifo.append(IDList[mi*seglen:(mi+1)*seglen])
        processfifo[-1] += IDList[processNum*seglen:]
        
        for patient1 in processfifo[processID]:
            patient1Folder = os.path.join(rootCTDir, patient1)
            cnt += 1
            oriCTDir = os.path.join(patient1Folder, patient1+'.mat')
            saveMaskName = os.path.join(patient1Folder, '%s_lungMask.mat' % (patient1))
            print('cnt', cnt, oriCTDir, saveMaskName)
            if os.path.exists(saveMaskName):
                print('dealed')
                continue
            img_data = sio.loadmat(oriCTDir)
            if 'new_img' in img_data.keys():
                img = img_data['new_img']
            elif 'new_img1' in img_data.keys():
                img = img_data['new_img1']
            if img.ndim == 4:
                img = img[:, :, :, 0]
            elif img.ndim == 5:
                img = img[:, :, :, 0, 0] 
            image = img.transpose(2,0,1)
            image[image<-1024] = -1024
            
            image = (image-image.mean(axis=(1,2), keepdims=True))/(image.std(axis=(1,2), keepdims=True) + 1e-6)
            batchSize = 64
            #ORIY, ORIX = image.shape[1:]
            #zoomSR = (1, ORIY/TARSIZE, ORIX/TARSIZE)#[y,x,chennel]
            imageFifo, lungpredFifo, lungMask = [], [], []
            for sidx in range(1, image.shape[0]-1):
                image1slice = image[sidx-1:sidx+2]
                image1slice = np.transpose(image1slice, axes=(1,2,0))
                zoomS = (TARSIZE/image1slice.shape[0], TARSIZE/image.shape[1], 1)#[y,x,chennel]
                image1slice = zoom(image1slice, zoom=zoomS, order=3)
                lungpredFifo.append(image1slice)
                if len(lungpredFifo) > batchSize:
                    lungpredFifo = np.asarray(lungpredFifo)
                    maskFifo = model.predict([lungpredFifo])
                    maskFifo = maskFifo[:,:,:,0]
                    #maskFifo = zoom(maskFifo, zoom=zoomSR, order=1)
                    #imageFifo.append(lungpredFifo[:,:,:,1])
                    lungMask.append(maskFifo)
                    lungpredFifo = []
            if len(lungpredFifo) > 0:
                lungpredFifo = np.asarray(lungpredFifo)
                maskFifo = model.predict([lungpredFifo])
                maskFifo = maskFifo[:,:,:,0]
                #maskFifo = zoom(maskFifo, zoom=zoomSR, order=1)
                #imageFifo.append(lungpredFifo[:,:,:,1])
                lungMask.append(maskFifo)
                lungpredFifo = []
            lungMask = np.concatenate(lungMask, axis=0)
            #lungMask[lungMask<0.5] = 0
            #lungMask[lungMask>=0.5] = 1
            self._writemat('lung_mask',lungMask.transpose(1,2,0), saveMaskName)
            print('image, mask', image.shape, lungMask.shape)
        print('cnt', cnt)
                
    def _writemat(self, key_name, mat, filename):
        scipy.io.savemat(filename, {key_name:mat})
        
    def _writenii(self, mat, filename, spacing=[0.75,0.75,5]):
        itkimage = sitk.GetImageFromArray(mat, isVector=False)
        itkimage.SetSpacing(spacing)
        #itkimage.SetOrigin(origin)
        sitk.WriteImage(itkimage, filename, True)
    
class RefineLungBox():
    def __init__(self):
        pass
        
    def extractLungArea(self, argvList):#the bounding cubic of lung
        argDic = {}
        for ar in argvList:
            k,v = ar.split('=')
            argDic[k] = v
        multiprocess = argDic['multiprocess']
        processID, processNum = multiprocess.split('/')
        processID = int(processID)
        processNum = int(processNum)
        assert processID < processNum
        
        dataset = argDic['dataset']
        dotMask = argDic['nonLungAreaSuppression']#reserve only the image inside lung
        if dotMask == 'false':
            dotMask = False
        elif dotMask == 'true':
            dotMask = True
            
        rootCTDir = dataset
        
        cntAll = 0
        LUNGAREATHRESH = 0.03
        TARSIZEy = 240.0
        TARSIZEx = 360.0
        TARDEPTH = 48.0

        allPatientfifo = os.listdir(rootCTDir)
        allPatientfifo = sorted(allPatientfifo)
        
        seglen = len(allPatientfifo)//processNum
        processfifo = []
        for mi in range(processNum):
            processfifo.append(allPatientfifo[mi*seglen:(mi+1)*seglen])
        processfifo[-1] += allPatientfifo[processNum*seglen:]
        total = len(processfifo[processID])
        print('allfile %d, current processID %d' % (len(allPatientfifo), processID))
        segModel = SegLung()
        for f1 in processfifo[processID]:
            f1Dir = os.path.join(rootCTDir, f1)
            
            ctDir =os.path.join(f1Dir, f1 + '.mat')
            maskDir = os.path.join(f1Dir,f1 + '_lungMask.mat')
            saveLungZoomDotMaskDir = os.path.join(f1Dir, f1+'_onlyLungZoomed3std.mat')

            print('ID %s %d/%d' % (f1, cntAll, total))
            
            if os.path.exists(saveLungZoomDotMaskDir):
                print('dealed, skip this case')
                continue
            
            img_data = sio.loadmat(ctDir)
            if 'new_img' in img_data.keys():
                img = img_data['new_img']
            elif 'new_img1' in img_data.keys():
                img = img_data['new_img1']
            if img.ndim == 4:
                img = img[:, :, :, 0]
            elif img.ndim == 5:
                img = img[:, :, :, 0, 0] 
            image = img.transpose(2,0,1)
            image = image[1:-1]
            image[image<-1024] = -1024#81x512x512, original -1024~1000+
            
            mask_data = sio.loadmat(maskDir)
            mask = mask_data['lung_mask']
            mask = mask.transpose(2,0,1)
            mask [mask<0.5] = 0
            mask [mask>=0.5] = 1
            MASKSCALE = (image.shape[1]/float(mask.shape[1]), image.shape[2]/float(mask.shape[2]))
            maskArea = mask.sum(axis=(1,2))
            ma = maskArea.max()
            areaRatio = maskArea / float(ma)
            
            areaRatio[areaRatio<LUNGAREATHRESH] = -1
            #check area non-continuous
            lungUse = []
            
            useThisSlice = []
            if (areaRatio[0] > 0) and (areaRatio[1] > 0):
                useThisSlice.append(0)
            for ai in range(1, areaRatio.shape[0]-1):
                if (areaRatio[ai-1] > 0) and (areaRatio[ai] > 0) and (areaRatio[ai+1] > 0):
                    useThisSlice.append(ai)
            if (areaRatio[-1] > 0) and (areaRatio[-2] > 0):
                useThisSlice.append(areaRatio.shape[0]-1)
            zstartUse = int(min(useThisSlice))
            zstopUse = int(max(useThisSlice)) + 1
            
            (_, ystart, xstart), (_, ystop, xstop) = self._boundingBox(mask)
            ystart *= MASKSCALE[0]
            ystop *= MASKSCALE[0]
            xstart *= MASKSCALE[1]
            xstop *= MASKSCALE[1]
            ystart = int(ystart)
            ystop = int(ystop)
            xstart = int(xstart)
            xstop = int(xstop)
            
            if (zstopUse-zstartUse < 3) or (ystop-ystart < 60) or (xstop-xstart < 60):
                print('lung mask is too small')
                continue
            lungUse = image[zstartUse:zstopUse, ystart:ystop, xstart:xstop]
            #######################non-lung area suppression######################
            if dotMask == True:
                maskUse = mask[zstartUse:zstopUse]
                maskUse = zoom(maskUse.astype(np.float32), (1, MASKSCALE[0], MASKSCALE[1]), order=1)
                maskUse = maskUse[:, ystart:ystop, xstart:xstop]
                
                full95 = np.percentile(lungUse, 95)
                lungUse[lungUse>full95] = full95
                lungarray = lungUse.copy()
                lungarray[maskUse<0.2] = 0
                lungarea = lungUse[maskUse>0.2]
                wallarray = lungUse.copy()
                lmean, lstd = lungarea.mean(), lungarea.std()
                #print('lmean', lmean-3*lstd, lmean+3*lstd)
                
                wallarray = np.clip(wallarray, lmean-4*lstd, lmean+4*lstd)
                wallarray[maskUse>0.2] = 0
                lungUse = lungarray + wallarray
            #######################non-lung area suppression######################
            luseM, luseS = lungUse.min(), lungUse.max()
            lungUseZoom = zoom(lungUse, (TARDEPTH/lungUse.shape[0], TARSIZEy/lungUse.shape[1], TARSIZEx/lungUse.shape[2]), order=3)
            lungUseZoom = np.clip(lungUseZoom, luseM, luseS)
            
            cntAll += 1
            print(lungUse.shape, lungUseZoom.shape)

            if dotMask == False:
                segModel._writemat('only_lung_zoomed', lungUseZoom.transpose(1,2,0), saveLungZoomDotMaskDir)
            else:
                segModel._writemat('only_lung_zoomed_3std', lungUseZoom.transpose(1,2,0), saveLungZoomDotMaskDir)
            
    def _boundingBox(self, A):
        B = np.argwhere(A)
        if A.ndim == 3:
            (zstart, ystart, xstart), (zstop, ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
            return (zstart, ystart, xstart), (zstop, ystop, xstop)
        elif A.ndim == 2:
            (ystart, xstart), (ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
            return (ystart, xstart), (ystop, xstop)
        else:
            print('box err')
            return
                
if __name__ == '__main__':
    
    ###############segment lung to acquire lung mask######################
    #if we use two processes to deal with the data in the folder, use these command lines:
    #python lungSeg.py dataset=../sampleData/validation1 gpu=0 multiprocess=0/2
    #python lungSeg.py dataset=../sampleData/validation1 gpu=1 multiprocess=1/2
    ######################################################################
    segmodel = SegLung()
    segmodel.pred(sys.argv[1:])
    
    ################extract lung area according to lung mask, and use non-lung area suppression######
    #if we use two processes to deal with the data in the folder, use these command lines:
    #python lungSeg.py dataset=../sampleData/validation1 nonLungAreaSuppression=true multiprocess=0/2
    #python lungSeg.py dataset=../sampleData/validation1 nonLungAreaSuppression=true multiprocess=1/2
    #################################################################################################
    #refine = RefineLungBox()
    #refine.extractLungArea(sys.argv[1:])
    
    print('haha')
    
    