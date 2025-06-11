import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class FFN(nn.Module):
    def __init__(self, dim, line_size,mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*line_size, dim*line_size,bias=False),
            GELU(),
            # nn.Linear(dim*mult, dim*line_size, bias=False),
            # GELU(),
            # nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return out

def ray_merge(x):

    line_bacth_num, line_size, c = x.shape
    point_batch = x.view(line_bacth_num , line_size*c)
    return point_batch
def ray_split(x,line_size):
    n,c = x.shape
    line_bacth = x.view(n , line_size, c//line_size)
    return line_bacth
class STAttention(nn.Module):
    def __init__(
            self,
            dim,
            line_size=4,
            dim_head=32,
            heads=4
    ):
        super().__init__()

        self.dim = dim  # dim = 输入维度 = 24
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        # position embedding
        seq_l = line_size  # 24
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, 4, 4))  # [1, 8, 24, 24]
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head  # 64 * 8
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # c -> inner_dim
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # c -> 2 * inner_dim
        self.to_out = nn.Linear(inner_dim, dim)  # inner_dim -> c

    def forward(self, x):
        """
        x: [n,c]
        return out: [n,c]
        n = N_ray * N_samples
        """
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2,dim=-1)

        q, k, v = map(
            lambda t: t.contiguous().view(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads).permute(0, 2, 1,3),(q, k, v))

        # scale
        q *= self.scale  # q / squart(d)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # Q, K 矩阵相乘
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # attn 和 v 相乘
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        return out
class ST_Attention_Blcok(nn.Module):
    def __init__(
            self,
            dim,
            split_size=24,
            dim_head=32,
            heads=8,
            num_blocks = 1
    ):
        super().__init__()
        self.split_size = split_size
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, STAttention(dim=dim,line_size=split_size,dim_head=dim_head,heads=heads)),
                PreNorm(dim*split_size, FFN(dim=dim,line_size=split_size))
            ]))

    def forward(self, x):
        """
        x: [n_ray*n_sample, c]
        return out: [n_ray*n_sample, c]
        """
        for (attn, ff) in self.blocks:
            x = ray_split(x,self.split_size)
            x = attn(x) + x
            x = ray_merge(x)
            x = ff(x)

        return x




if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ST_Attention_Blcok(dim=32,heads=1,dim_head=32,line_size=4).to(device)
    begin = time.time()
    inputs = torch.randn((409600, 128)).to(device)
    a = model(inputs)
    end = time.time()
    print(end-begin)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')