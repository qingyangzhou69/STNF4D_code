import cv2
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
import pydicom
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
def convert_to_attenuation(data: np.array, rescale_slope=1, rescale_intercept=0):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu
def convert_to_HU(mu):
    mu_water = 0.206
    mu_air = 0.0004
    HU = (mu-mu_water)*1000/(mu_water - mu_air)
    return HU

def norml(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
def read_GT(path):
    # num_slice=142

    volumes = []
    for i in range(10):
        file_path = path+ '/Phase'+str(i + 1)+'/'
        files = os.listdir( file_path)
        files.sort(key=lambda x: int(x[2:-4]))

        volume = []
        for j in range(len(files)):
            # image = cv2.imread(file_path+files[j],0)
            # image = pydicom.dcmread(file_path+files[j])
            image = itk.ReadImage(file_path + files[j])
            spacing = image.GetSpacing()
            pixel_array = itk.GetArrayFromImage(image)
            volume.append(pixel_array[0])
        volumes.append(volume)
    volumes = np.array(volumes)
    volumes = np.clip(volumes, -1000, 1000)   ###cut HU范围
    # volumes = (norml(volumes)*1500-1000).astype(np.float32)
    volumes = convert_to_attenuation(volumes).astype(np.float32)
    volumes = np.transpose(volumes,(0,2,3,1))
    volumes = volumes[...,64:]  #cut关键 101为64: 其他为30:-30
    volumes = volumes[:,163:450,66:470, :] #100HM: 113:400,66:470  101HM:450,66:470  102HM 163:450,66:470   103HM 163:450,66:470
    volumes = np.flip(volumes, axis=-1)
    return volumes

if __name__ == '__main__':
    case = '101_HM'
    GTpath = './4Dlung/'+case
    configPath = './config.yml'

    img = read_GT(GTpath)

    with open(configPath, 'r') as handle:
        data = yaml.safe_load(handle)


    data['image'] = img
    data['nVoxel'] = [img.shape[1], img.shape[2], img.shape[3]]
    geo = ConeGeometry_special(data)
    num_phase = 10
    train_scales = 10
    # outputPath = './data/' + case + str(num_phase*train_scales)+'wooff_cuthu'+'.pickle' ###
    outputPath = '../../data/' + case +'.pickle' ###
    val_scales = 5
    train_phase = np.linspace(1, num_phase, num_phase)
    train_phase = np.tile(train_phase, [train_scales, 1])
    train_phase = train_phase.reshape(-1).astype(np.int32)
    data['train']={'phase':train_phase}
    val_phase = np.linspace(1, num_phase, num_phase)
    val_phase = np.tile(val_phase, [val_scales, 1])
    val_phase = val_phase.reshape(-1).astype(np.int32)
    data['val']={'phase':val_phase}

    data['numTrain'] = num_phase*train_scales
    data['numVal'] = num_phase*val_scales

    data['train']['angles'] = np.linspace(0, 360 / 180 * np.pi, data['numTrain'] + 1)[:-1] + data['startAngle'] / 180 * np.pi
    projections=[]
    for i in range(num_phase):
        project=tigre.Ax(np.transpose(img[i], (2, 1, 0)).copy(), geo, data['train']['angles'])[:, ::-1, :]
        projections.append(project)
    data['train']['projections'] = np.zeros_like(projections[0])
    projections = np.array(projections).astype(np.float32)
    for i in range(data['numTrain']):
        data['train']['projections'][i] = projections[train_phase[i]-1][i]

    data['val']['angles'] = np.linspace(0, 360 / 180 * np.pi, data['numVal'] + 1)[:-1] + data['startAngle'] / 180 * np.pi
    projections=[]
    for i in range(num_phase):
        project=tigre.Ax(np.transpose(img[i], (2, 1, 0)).copy(), geo, data['val']['angles'])[:, ::-1, :]
        projections.append(project)
    data['val']['projections'] = np.zeros_like(projections[0])
    projections = np.array(projections).astype(np.float32)
    for i in range(data['numVal']):
        data['val']['projections'][i] = projections[val_phase[i]-1][i]


    print('Display ct image')
    tigre.plotimg(img[0].transpose((2,0,1)), dim='z')
    print('Display training images')
    tigre.plotproj(data['train']['projections'][:, ::-1, :])
    print('Display validation images')
    tigre.plotproj(data['val']['projections'][:, ::-1, :])
    # Save data
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with open(outputPath, 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"Save files in {outputPath}")
