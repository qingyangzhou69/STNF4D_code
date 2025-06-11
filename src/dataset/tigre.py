import torch
import pickle
import sys
import numpy as np

from torch.utils.data import DataLoader, Dataset


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """

    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data['DSD']/1000 # Distance Source Detector      (m)
        self.DSO = data['DSO']/1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(
            [data['nDetector'][1], data['nDetector'][0]])  # number of pixels              (px)
        self.dDetector = np.array(data['dDetector'])/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data['nVoxel'])  # number of voxels              (vx)
        self.dVoxel = np.array(data['dVoxel'])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data['offOrigin'])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data['offDetector'][0], data['offDetector'][1], 0]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data['accuracy']  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data['mode']  # parallel, cone                ...
        self.filter = data['filter']



class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, is_dynet=False ,n_rays=1024, type='train', device='cuda'):
        super().__init__()

        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        self.device = device
        self.geo = ConeGeometry(data)
        # self.geo.dVoxel= self.geo.dVoxel*2
        # self.geo.sVoxel = self.geo.sVoxel*2
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)
        self.s1,self.s2,self.s3=1/(self.geo.sVoxel / 2 - self.geo.dVoxel / 2)
        self.bound = self.geo.sVoxel / 2 - self.geo.dVoxel / 2
        self.xyz_min = -self.geo.sVoxel / 2
        self.xyz_max = self.geo.sVoxel / 2
        if type == 'train':
            self.projs = data['train']['projections']
            angles = data['train']['angles']
            rays ,self.poses= self.get_rays(angles, self.geo, device)
            self.rays = np.concatenate([rays, np.ones_like(rays[...,:1])*self.near, np.ones_like(rays[...,:1])*self.far], axis=-1)
            self.n_samples = data['numTrain']
            if is_dynet==True:
                self.phase= data['train']['phase']
            self.images_idx = np.linspace(0, self.n_samples - 1, self.n_samples)

            self.voxels = self.get_voxels(self.geo)
        elif type == 'val':
            self.projs = data['val']['projections']
            angles = data['val']['angles']
            rays,self.poses = self.get_rays(angles, self.geo, device)
            self.n_samples = angles.shape[0]
            self.rays = np.concatenate([rays, np.ones_like(rays[...,:1])*self.near, np.ones_like(rays[...,:1])*self.far], axis=-1)

            if is_dynet == True:
                self.phase = data['val']['phase']
            # self.image = data['image']
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=self.device)
            self.image = data['image'][0][20:60]
            self.voxels = self.voxels[20:60] #100:107
        elif type == 'demo':
            # self.projs = data['val']['projections']
            # angles = data['val']['angles']
            if is_dynet == True:
                self.phase = data['val']['phase']
            # rays,self.poses = self.get_rays(angles, self.geo, device)
            # self.n_samples = angles.shape[0]
            # self.rays = np.concatenate([rays, np.ones_like(rays[...,:1])*self.near, np.ones_like(rays[...,:1])*self.far], axis=-1)
            # self.rays = torch.tensor(self.rays, dtype=torch.float32, device=self.device)
            # self.n_samples = data['numVal']
            self.image = data['image']
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=self.device)


        





    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing='ij')
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    
    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        poses = []
        for angle in angles:
            pose = self.angle2pose(geo.DSO, angle)
            rays_o, rays_d = None, None
            if geo.mode == 'cone':
                i, j = np.meshgrid(np.linspace(0, W - 1, W),
                                    np.linspace(0, H - 1, H))   # pytorch's meshgrid has indexing='ij'
                uu = (i + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = np.stack([uu / DSD, vv / DSD, np.ones_like(uu)], -1)
                rays_d = np.sum(np.matmul(pose[:3,:3], dirs[..., None]), -1) # pose[:3, :3] *
                rays_o = np.tile(pose[:3, -1],(rays_d.shape[0],rays_d.shape[1],1))
                # import open3d as o3d
                # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
                # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                # poseray = plot_camera_pose(pose.cpu().detach().numpy())
                # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            elif geo.mode == 'parallel':
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device))  # pytorch's meshgrid has indexing='ij'
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)

            # import open3d as o3d
            # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
            # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
            # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
            # rays1 = plot_rays(rays_d, rays_o, 2)
            # poseray = plot_camera_pose(pose)
            # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            
            # else:
            #     NotImplementedError('Unknown CT scanner type!')
            rays.append(np.concatenate([rays_o, rays_d], axis=-1))
            poses.append(pose)
        


        return np.stack(rays, axis=0),np.stack(poses, axis=0)
    

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    
    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
