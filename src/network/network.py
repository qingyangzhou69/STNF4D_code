import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .transformer import *
def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)
def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim





def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) /(aabb[1] - aabb[0]+0.00001)


class DynamicNetwork(nn.Module):
    def __init__(self, encoder,bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation='sigmoid',aabb=0.2, num_phases=10):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.num_phases = num_phases
        self.in_dim = self.encoder.n_output_dims
        self.bound = bound
        self.aabb=aabb

        self.hash_encoder_size=4
        self.STFormer = ST_Attention_Blcok(dim=self.in_dim//self.hash_encoder_size,
                                             split_size=self.hash_encoder_size ,
                                             dim_head=self.in_dim//self.hash_encoder_size,
                                             heads=1,
                                             num_blocks=1)
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                                    else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in
                                                    range(1, num_layers - 1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.Softplus() for i in range(0, num_layers - 1, 1)])
        if last_activation == 'sigmoid':
            self.activations.append(nn.Sigmoid())
        elif last_activation == 'relu':
            self.activations.append(nn.ReLU())
        elif last_activation == 'softplus':
            self.activations.append(nn.Softplus())
        else:
            NotImplementedError('Unknown last activation')

    def forward(self, x):

        t = (x[..., -1] - 1) / (self.num_phases-1)
        pts = (x[..., :3] + self.bound) / (2 * self.bound)

        x = self.encoder(pts,t)##

        x = self.STFormer(x)
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
            if i == len(self.layers) - 1:
                h = x
            x = linear(x)
            x = activation(x)

        return x
    