import os
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import time
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as functional
import argparse
from PIL import Image
import imageio
from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer,Trainer_val
from src.loss import calc_mse_loss
from src.util import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image,plot_flow
# from src import plot_flow
from torch.nn.functional import grid_sample
# from dataGenerator.GenerateTrueData import convert_to_attenuation,convert_to_hu
#生成所有phase结果
def convert_to_HU(mu):
    mu_water = 0.206
    mu_air = 0.0004
    HU = (mu-mu_water)*1000/(mu_water - mu_air)
    return HU
def windowsee8bit(HU,wind):

    clip_hu = np.clip(HU, wind[0], wind[1], out=None)
    norm = (clip_hu-wind[0])/(wind[1]-wind[0])
    out = (255*norm).astype(np.uint8)
    return out
def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/abdomen_50.yaml',
                        help='configs file path')
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)
def to8bit(x):
    _range = np.max(x) - np.min(x)
    norm = (x - np.min(x)) / _range
    out = (255*norm).astype(np.uint8)
    return out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class demo(Trainer_val):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")
    def compute_loss(self, data, global_step, idx_epoch):
        rays = data['rays'].reshape(-1, 8)
        projs = data['projs'].reshape(-1)
        ret = render(rays, self.net, self.net_fine, **self.conf['render'])
        projs_pred = ret['acc']

        loss = {'loss': 0.}
        calc_mse_loss(loss, projs, projs_pred)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f'train/{ls}', loss[ls].item(), global_step)

        return loss['loss']
    def output(self):



        SSIM = []
        PSNR = []
        for i in range(10):
            phase_num=i+1
            voxel_path = self.expdir + '/demo/' + 'voxelallphase/'
            if not os.path.exists( voxel_path):
                os.makedirs( voxel_path)

            image = self.eval_dset.image[i]
            phase = torch.ones(image.shape).unsqueeze(-1).to(self.device)*phase_num
            voxels = torch.tensor(self.eval_dset.voxels, dtype=torch.float32, device=self.device)

            comb = torch.cat([voxels, phase], dim=-1)

            image_pred = run_network(comb, self.dy_net,self.netchunk)
            image_pred = image_pred.squeeze()
            image_pred_np=image_pred.detach().cpu().numpy()
            # SSIM.append(image_pred, image)
            # PSNR =

            loss = {

                'psnr_3d': get_psnr_3d(image_pred, image),
                'ssim_3d': get_ssim_3d(image_pred, image),
            }
            image_pred_np = convert_to_HU(image_pred_np)
            if not os.path.exists( voxel_path+'Phase'+str(phase_num)+'/'):
                os.makedirs( voxel_path+'Phase'+str(phase_num)+'/')
            # for i in range(image_pred_np.shape[2]):
            #     index = str(i)
            #     (Image.fromarray(windowsee8bit(image_pred_np[:, :, i],[-1000,500]))).save(voxel_path+'Phase'+str(phase_num)+'/'+index + '.png')
            # print(f'[EVAL] {self.fmt_loss_str(loss)}')
            PSNR.append(loss['psnr_3d'])
            SSIM.append(loss['ssim_3d'])
        PSNR = np.array(PSNR)
        SSIM = np.array(SSIM)

        print('PSNRmean:%.2f,PSNRtsd:%.2f' % (PSNR.mean(), PSNR.std()))
        print('SSIMmean:%.4f,SSIMtsd:%.4f' % (SSIM.mean(), SSIM.std()))

    def run_test(self):
        self.dy_net.eval()

        with torch.no_grad():
            self.output()
demoer=demo()
demoer.run_test()


