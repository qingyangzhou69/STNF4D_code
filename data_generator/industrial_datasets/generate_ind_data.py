import os
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
from scipy import signal
import pickle
import scipy.io
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import random
import SimpleITK as itk
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import math

import torch.nn.functional as functional
import torch


def inter(image,size):
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    image = functional.interpolate(image, size=size, mode='nearest')
    image = image.squeeze(0).squeeze(0).numpy()
    return image
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data['DSD'] / 1000  # Distance Source Detector      (m)
        self.DSO = data['DSO'] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data['nDetector'])  # number of pixels              (px)
        self.dDetector = np.array(data['dDetector']) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data['nVoxel'][::-1])  # number of voxels              (vx)
        self.dVoxel = np.array(data['dVoxel'][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data['offOrigin'][::-1]) / 1000  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data['offDetector'][1], data['offDetector'][0], 0]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data['accuracy']  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data['mode']  # parallel, cone                ...
        self.filter = data['filter']
def loadimage(path,phase_num=10):
    images = []
    for i in range(phase_num):
        impath = path+"%.2d"%(i)+'.npy'
        # image = np.fromfile(impath, dtype=np.float32)
        image = np.load(impath)
        image = np.array(image).reshape(256,256,256)
        # torch.tensor(image)
        # image = inter(image,[256,256,256])
        # Interpolation3D(image,(256,256,256))
        images.append(image)
    images = np.array(images)

    return images
if __name__ == '__main__':

    S='12'
    m='021'


    path = '2563^LLNL/D4DCT_DFM/S'+S+'_'+m+'/'+'00_256_180_180_gt_f0'
    configPath = './config.yml'
    # path = '803^LLNL/bb1507819f_3_1/D4DCT_DFM/S00_101/00_080_720_090_gt_f009.raw'
    images = loadimage(path,60)

    outputPath = '../../data/S'+S+'_'+m+'_256_60.pickle'

    with open(configPath, 'r') as handle:
        data = yaml.safe_load(handle)


    data['image'] = images
    data['nVoxel'] = [images.shape[1], images.shape[2], images.shape[3]]
    geo = ConeGeometry_special(data)
    num_phase = 60
    train_scales = 4
    val_scales = 1
    data['numphase'] =num_phase
    train_phase = np.linspace(1, num_phase, num_phase)
    train_phase = np.tile(train_phase, [train_scales, 1])
    train_phase = np.swapaxes(train_phase, 0, 1)
    train_phase = train_phase.reshape(-1).astype(np.int32)
    data['train'] = {'phase': train_phase}

    val_phase = np.linspace(1, num_phase, num_phase)
    val_phase = np.tile(val_phase, [val_scales, 1])
    val_phase = np.swapaxes(val_phase, 0, 1)
    val_phase = val_phase.reshape(-1).astype(np.int32)
    data['val']={'phase':val_phase}

    data['numTrain'] = num_phase*train_scales
    data['numVal'] = num_phase*val_scales

    data['train']['angles'] = np.linspace(0, 180 / 180 * np.pi, data['numTrain'] + 1)[:-1] + data['startAngle'] / 180 * np.pi
    projections=[]
    for i in range(num_phase):
        project=tigre.Ax(np.transpose(images[i], (2, 1, 0)).copy(), geo, data['train']['angles'])[:, ::-1, :]
        projections.append(project)
    data['train']['projections'] = np.zeros_like(projections[0])
    projections = np.array(projections).astype(np.float32)
    for i in range(data['numTrain']):
        data['train']['projections'][i] = projections[train_phase[i]-1][i]

    data['val']['angles'] = np.linspace(0, 360 / 180 * np.pi, data['numVal'] + 1)[:-1] + data['startAngle'] / 180 * np.pi
    projections=[]
    for i in range(num_phase):
        project=tigre.Ax(np.transpose(images[i], (2, 1, 0)).copy(), geo, data['val']['angles'])[:, ::-1, :]
        projections.append(project)
    data['val']['projections'] = np.zeros_like(projections[0])
    projections = np.array(projections).astype(np.float32)
    for i in range(data['numVal']):
        data['val']['projections'][i] = projections[val_phase[i]-1][i]
    a=0

    print('Display ct image')
    tigre.plotimg(images[0].transpose((2,0,1)), dim='z')
    print('Display training images')
    tigre.plotproj(data['train']['projections'][:, ::-1, :])
    # print('Display validation images')
    tigre.plotproj(data['val']['projections'][:, ::-1, :])

    # Save data
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with open(outputPath, 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    # (Image.fromarray(projs[0,0, :, :])).save('projshow.tif')