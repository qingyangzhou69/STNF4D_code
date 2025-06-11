import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
# import tinycudann as tcnn
from .backend import _backend
import math
def reduction_func(reduction):
    # x: list of Tensor
    if reduction == 'prod':
        return math.prod
    elif reduction == 'sum':
        return sum
    elif reduction == 'mean':
        return lambda x: sum(x)/len(x)
    elif reduction == 'concat':
        return lambda x: torch.cat(x, dim=-1)
    else:
        raise ValueError("Invalid reduction")
class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    #@custom_fwd
    def forward(ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        H = base_resolution # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, L * C]

        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None
        else:
            return None, grad_embeddings, None, None, None


hash_encode = _hash_encode.apply



class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.n_output_dims = num_levels * level_dim
        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = base_resolution * 2 ** i
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim) # limit max number
            #params_in_level = int(params_in_level / 8) * 8 # make divisible
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))
        
        self.n_params = self.offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"
    
    def forward(self, inputs):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        # if inputs.min().item() < -size or inputs.max().item() > size:
        #     raise ValueError(f'HashGrid encoder: inputs range [{inputs.min().item()}, {inputs.max().item()}] not in [{-size}, {size}]!')


        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = hash_encode(inputs, self.embeddings, self.offsets, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs




class HashGrid4D(nn.Module):
    def __init__(
            self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19,
            hash_size_dynamic=[15, 13, 13],  # larger for xy
            decompose=False,  # static & dynamic
            reduction='concat',  # 'concat'/'prod'/'sum'/'mean'
    ):
        super().__init__()
        # xyz
        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.hash_static = HashEncoder(self.input_dim,self.num_levels,self.level_dim, self.base_resolution, self.log2_hashmap_size)

        # xyt, xzt, yzt
        hash_dynamic = []
        # for i in range(3):
        #     hash_dynamic.append(
        #         HashGridT(
        #             time_resolution=10,
        #             base_resolution=512,
        #             max_resolution=32768,
        #             n_levels=8,
        #             n_features_per_level=2,
        #             log2_hashmap_size=hash_size_dynamic[i],
        #         )
        #     )
        for i in range(3):
            hash_dynamic.append(
                HashEncoder(
                    self.input_dim, self.num_levels, self.level_dim, self.base_resolution, self.log2_hashmap_size,
                )
            )
        self.hash_dynamic = nn.ModuleList(hash_dynamic)

        self.decompose = decompose
        self.reduction = reduction
        if reduction == 'concat':
            self.n_output_dims = self.hash_static.n_output_dims + self.hash_dynamic[0].n_output_dims * 3
        else:
            self.n_output_dims = self.hash_static.n_output_dims + self.hash_dynamic[0].n_output_dims

    def forward_static(self, x):
        hash_feat_static = self.hash_static(x)  # xyz grid

        return hash_feat_static

    def forward_dynamic(self, x, t):
        xy = torch.cat([x[:, [0, 1]],t.unsqueeze(1)],axis=-1)
        xz = torch.cat([x[:, [0, 2]],t.unsqueeze(1)],axis=-1)
        yz = torch.cat([x[:, [1, 2]],t.unsqueeze(1)],axis=-1)

        hash_feat_xyt = self.hash_dynamic[0](xy)  # xyt grid
        hash_feat_xzt = self.hash_dynamic[1](xz)  # xzt grid
        hash_feat_yzt = self.hash_dynamic[2](yz)  # yzt grid

        reduction = reduction_func(self.reduction)
        hash_feat_dynamic = reduction([hash_feat_xyt, hash_feat_xzt, hash_feat_yzt])

        return hash_feat_dynamic

    def forward(self, x, t):
        ## x.shape: [N, 3], in [0, 1]
        ## t: float, in [0, 1]


        hash_feat_static = self.forward_static(x)

        hash_feat_dynamic = self.forward_dynamic(x, t)

        if self.decompose:
            hash_feat = [hash_feat_static, hash_feat_dynamic]
        else:
            hash_feat = torch.cat([hash_feat_static, hash_feat_dynamic], dim=-1)

        return hash_feat