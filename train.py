import os
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import time

import argparse

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import *
from src.util import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/abdomen_50.yaml',
                        help='configs file path')
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")
    def compute_loss_ori(self, rays,projs,global_step, idx_epoch):
        # rays = rays.reshape(-1, 8)
        # projs = projs.reshape(-1)
        ret = render(rays, self.net, self.net_fine, **self.conf['render'])
        projs_pred = ret['acc']

        loss = {'loss': 0.}

        calc_mse_loss(loss, projs, projs_pred)
        loss['loss'] +=self.s3im_func(projs, projs_pred)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f'train/{ls}', loss[ls].item(), global_step)

        return loss['loss']

    def compute_loss(self, projs_pred, projs, extra_loss, global_step, idx_epoch,img_pred=None):
        # rays = data['rays'].reshape(-1, 8)
        # projs = data['projs'].reshape(-1)
        # ret = render(rays, self.net, self.net_fine, **self.conf['render'])
        # projs_pred = ret['acc']

        loss = {'loss': 0.}


        calc_mse_loss(loss, projs, projs_pred)
        if extra_loss!=0:
            loss['extra_loss'] = extra_loss
            loss['loss']+=extra_loss*0.001
        if img_pred!=None:
            img_pred=img_pred.squeeze()
            calc_tv_loss_t(loss,img_pred,0.000001)
            calc_La_loss_t(loss, img_pred, 0.000001)
        # loss['loss'] += self.s3im_func(projs, projs_pred)
        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f'train/{ls}', loss[ls].item(), global_step)

        return loss


    def eval_step(self, global_step):
        """
        Evaluation step
        """
        # Evaluate projection
        select_ind = np.random.choice(self.eval_dset.n_samples)
        projs = self.eval_dset.projs[select_ind]
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8)

        phase = self.eval_dset.phase[select_ind].reshape((1, 1))
        phase = np.tile(phase,[1024,1])
        phase = torch.tensor(phase, dtype=torch.float32, device=self.device)

        projs = torch.tensor(projs, dtype=torch.float32, device=self.device)
        rays = torch.tensor(rays, dtype=torch.float32, device=self.device)

        H, W = projs.shape
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(render(rays[i:i+self.n_rays],phase, self.dy_net, self.dy_net, **self.conf['render'])['acc'])
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W)

        # Evaluate density
        image = self.eval_dset.image
        phase = torch.ones(image.shape).unsqueeze(-1).to(self.device)
        comb = torch.cat([self.eval_dset.voxels, phase], dim=-1)
        image_pred = run_network(comb, self.dy_net, self.netchunk)
        # image_pred = run_network(self.eval_dset.voxels, self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            'proj_mse': get_mse(projs_pred, projs),
            'proj_psnr': get_psnr(projs_pred, projs),
            'psnr_3d': get_psnr_3d(image_pred, image),
            'ssim_3d': get_ssim_3d(image_pred, image),
        }

        # Logging
        show_slice = 5
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image = torch.tensor(show_image, dtype=torch.float32, device=self.device)
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image_pred[..., i_show], show_image[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        show_proj = torch.concat([projs, projs_pred], dim=1)
        #
        self.writer.add_image('eval/density', cast_to_image(show_density), global_step, dataformats='HWC')
        self.writer.add_image('eval/projection', cast_to_image(show_proj), global_step, dataformats='HWC')

        for ls in loss.keys():
            self.writer.add_scalar(f'eval/{ls}', loss[ls], global_step)

        return loss


trainer = BasicTrainer()
trainer.start()

