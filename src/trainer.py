import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
import random
from .dataset import TIGREDataset as Dataset
from .render import render, run_network
from .network import get_network
from .encoder import get_encoder
from .loss import *
from .util import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image,AverageMeter
import math
# import tinycudann as tcnn
class Trainer:
    def __init__(self, cfg, device='cuda'):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg['render']['n_fine']
        self.epochs = cfg['train']['epoch']
        self.i_eval = cfg['log']['i_eval']
        self.i_save = cfg['log']['i_save']
        self.netchunk = cfg['render']['netchunk']
        self.n_rays = cfg['train']['n_rays']
        self.is_dynet = cfg['is_dynet']
        self.num_phases = cfg['num_phases']

        # Log direcotry
        self.expdir = os.path.join(cfg['exp']['expdir'], cfg['exp']['expname'])
        self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
        self.demodir = os.path.join(self.expdir, 'demo')
        os.makedirs(self.demodir, exist_ok=True)

        # Dataset
        self.train_dset = Dataset(cfg['exp']['datadir'],cfg['is_dynet'], cfg['train']['n_rays'], 'train', device)
        self.eval_dset = Dataset(cfg['exp']['datadir'],cfg['is_dynet'], cfg['train']['n_rays'], 'val', device) if self.i_eval > 0 else None
        # self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg['train']['n_batch'])
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None
    
        # Network
        self.device = device
        bbox = torch.tensor(np.stack((self.train_dset.xyz_min,self.train_dset.xyz_max),axis=-1).transpose(1,0).astype(np.float32)).cuda()
        dynamecnet = get_network(cfg['dy_network']['net_type'])
        cfg['dy_network'].pop('net_type', None)
        encoder = get_encoder(**cfg['dy_encoder'])
        self.dy_net = dynamecnet(encoder,num_phases=self.num_phases,aabb=bbox,**cfg['dy_network']).to(device)
        grad_vars = list(self.dy_net.parameters())





        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.s3im_func = S3IM(kernel_size=4, stride=4, repeat_time=10,
        #                  patch_height=32, patch_width=32).cuda()
        # Optimizer
        # self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg['train']['lrate'], betas=(0.9, 0.999))

        # Load checkpoints
        self.epoch_start = 0
        if cfg['train']['resume'] and os.path.exists(self.ckptdir):
            print(f'Load checkpoints from {self.ckptdir}.')
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt['epoch'] + 1
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_step = self.epoch_start
            self.dy_net.load_state_dict(ckpt['dy_network'])


        self.optimizer = torch.optim.Adam( grad_vars, lr=cfg['train']['lrate'], betas=(0.9, 0.999))
        # self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg['train']['lrate'], betas=(0.9, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                            T_max= cfg['train']['epoch'],
                                                            eta_min=1e-6)
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=2000,
                                                            num_training_steps=cfg['train']['epoch'],
                                                            eta_min=1e-6
                                                            )
        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text('parameters', self.args2string(cfg), global_step=0)


    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))


    def start(self):
        """
        Main loop.
        """
        train_datas = {}
        avg_loss = AverageMeter()
        def fmt_loss_str(losses):
            return ''.join(', ' + k + ': ' + f'{losses[k].item():.3g}' for k in losses)

        avg_loss_tv = AverageMeter()

        N_rand = self.n_rays

        #加载射线信息
        rays = self.train_dset.rays.reshape(-1,8)
        rays = rays.astype(np.float32)
        imagesf = self.train_dset.projs
        num_img,H,W= imagesf.shape
        train_datas['rays'] = rays
        train_datas['projs'] = imagesf.reshape(-1, 1)
        # 加载相位信息
        phase_tile=self.train_dset.phase.reshape((num_img, 1, 1)) ###
        phase_tile=np.tile(phase_tile, [1, H, W])
        train_datas['phase']=phase_tile.reshape(-1, 1).astype(np.int64)


        images_idx_tile = self.train_dset.images_idx.reshape((num_img, 1, 1))
        images_idx_tile = np.tile(images_idx_tile, [1, H, W])
        train_datas['images_idx'] = images_idx_tile.reshape(-1, 1).astype(np.int64)

        print('shuffle rays')
        shuffle_idx = np.random.permutation(len(train_datas['rays']))
        train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
        # train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}
        print('done')

        i_batch = 0
        #开始循环
        pbar=tqdm(total = self.epochs)
        for idx_epoch in range(self.epoch_start, self.epochs): # trange(self.idx_epoch_start, self.epochs, desc='epoch'):

            self.global_step += 1

            iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()}
            iter_data = {k: torch.tensor(v).cuda() for k, v in iter_data.items()}
            i_batch += N_rand
            if i_batch >= len(train_datas['rays']):
                print("Shuffle data after an epoch!")
                shuffle_idx = np.random.permutation(len(train_datas['rays']))
                train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}

                i_batch = 0

            # Train
            self.dy_net.train()
            extra_loss = 0
            ret = render(iter_data['rays'],iter_data['phase'], self.dy_net, self.dy_net, **self.conf['render'])

            pred_proj = ret['acc']

            image_pred = None
            if self.conf['train']['isTVloss'] and idx_epoch>=6000:
                x,y,z,_ = self.train_dset.voxels.shape
                xdim = 64
                ydim = 64
                zdim=5
                tdim=3
                randz = random.randint(0,z-zdim-1)
                randx = random.randint(0, x - xdim - 1)
                randy = random.randint(0, y - ydim - 1)
                randt = random.randint(1,self.num_phases-2)

                voxels = torch.tensor(self.train_dset.voxels[randx:randx+xdim,randy:randy+ydim,randz:randz+zdim,:], dtype=torch.float32, device=self.device)
                phase_pre = (torch.ones([xdim,ydim,zdim,1]) * randt).cuda()
                phase_next = (torch.ones([xdim,ydim,zdim,1]) * (randt + 1)).cuda()
                phase_next2 = (torch.ones([xdim,ydim,zdim,1])*(randt+2) ).cuda()
                #
                comb = torch.cat([voxels, phase_pre], dim=-1)
                warp_pre = run_network(comb, self.dy_net, self.netchunk)
                #
                comb = torch.cat([voxels, phase_next], dim=-1)
                warp_next = run_network(comb, self.dy_net, self.netchunk)

                comb = torch.cat([voxels, phase_next2], dim=-1)
                warp_next2 = run_network(comb, self.dy_net, self.netchunk)
                image_pred = torch.stack([warp_pre,warp_next,warp_next2],dim=0)


            self.optimizer.zero_grad()
            loss_train = self.compute_loss(pred_proj, iter_data['projs'].reshape(-1), extra_loss, global_step=self.global_step, idx_epoch=idx_epoch,img_pred=image_pred)

            loss_train['loss'].backward()
            self.optimizer.step()

            avg_loss.update(loss_train['loss'].item(),1)


            pbar.set_postfix({'loss': avg_loss.avg,#'loss_tv':avg_loss_tv.avg,
                              'lr': self.optimizer.param_groups[0]['lr']})
            pbar.update(1)

            # Evaluate
            if idx_epoch % self.i_eval == 0 and self.i_eval > 0:
                self.dy_net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step)
                self.dy_net.train()
                tqdm.write(f'[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}')
            
            # Save
            if (idx_epoch+1) % self.i_save == 0 or idx_epoch == self.epochs-1:
                if os.path.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                path = os.path.join(self.expdir, str(idx_epoch)+"_ckpt.tar")
                tqdm.write(f'[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}')
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "dy_network":self.dy_net.state_dict() if self.is_dynet else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        print(f'Training complete! {self.expdir}')

    def train_step(self, rays,projs, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss_ori(rays,projs, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def compute_loss(self, projs_pred, projs, align_loss, global_step, idx_epoch,):
        """
        Training step
        """
        raise NotImplementedError()


    def eval_step(self, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()


class Trainer_val:
    def __init__(self, cfg, device='cuda',num_view=None):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.device = device
        self.n_fine = cfg['render']['n_fine']
        self.epochs = cfg['train']['epoch']
        self.i_eval = cfg['log']['i_eval']
        self.i_save = cfg['log']['i_save']
        self.netchunk = cfg['render']['netchunk']
        self.n_rays = cfg['train']['n_rays']
        self.is_dynet = cfg['is_dynet']
        self.num_phases = cfg['num_phases']
        # Log direcotry

        self.expdir = os.path.join(cfg['exp']['expdir'], cfg['exp']['expname'])

        self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
        self.demodir = os.path.join(self.expdir, 'demo')
        os.makedirs(self.demodir, exist_ok=True)

        # Dataset
        # self.train_dset = Dataset(cfg['exp']['datadir'],cfg['is_dynet'], cfg['train']['n_rays'], 'train', device)
        self.eval_dset = Dataset(cfg['exp']['datadir'], cfg['is_dynet'],cfg['train']['n_rays'], 'demo', device)

        self.eval_dloader = torch.utils.data.DataLoader(self.eval_dset, batch_size=cfg['train']['n_batch'])

        # Network

        bbox = torch.tensor(np.stack((self.eval_dset.xyz_min,self.eval_dset.xyz_max),axis=-1).transpose(1,0).astype(np.float32)).cuda()

        dynamecnet = get_network(cfg['dy_network']['net_type'])
        cfg['dy_network'].pop('net_type', None)
        encoder = get_encoder(**cfg['dy_encoder'])
        self.dy_net = dynamecnet(encoder,num_phases=self.num_phases,aabb=bbox,**cfg['dy_network']).to(device)
        grad_vars = list(self.dy_net.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg['train']['lrate'], betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=cfg['train']['lrate_gamma'])
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
        #                                                     step_size=cfg['train']['lrate_step'],
        #                                                     gamma=cfg['train']['lrate_gamma'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                            T_max= cfg['train']['epoch']*150,
                                                            eta_min=1e-6)
        # Load checkpoints
        self.epoch_start = 0
        if cfg['train']['resume'] and os.path.exists(self.ckptdir):
            print(f'Load checkpoints from {self.ckptdir}.')
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt['epoch'] + 1
            # self.optimizer.load_state_dict(ckpt['optimizer'],strict=False)
            # self.global_step = self.epoch_start * len(self.train_dloader)
            self.dy_net.load_state_dict(ckpt['dy_network'])

    def fmt_loss_str(self,losses):
        return ''.join(', ' + k + ': ' + f'{losses[k].item():.4g}' for k in losses)
    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))


    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        eta_min: float = 0.0,
        num_cycles: float = 0.999,
        last_epoch: int = -1,
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)