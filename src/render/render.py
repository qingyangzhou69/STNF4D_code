import torch
import torch.nn as nn

def render(rays, phase,net, dy_net, n_samples, n_fine, perturb, netchunk, raw_noise_std,deepth=None,useAWP=False):

    n_rays = rays.shape[0]
    rays_o, rays_d, near, far = rays[...,:3], rays[...,3:6], rays[...,6:7], rays[...,7:]
    if deepth!=None:
        deepth = deepth.reshape(near.shape)
        near = near-deepth
        far = far-deepth
    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.expand([n_rays, n_samples])

    if perturb:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=lower.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
    bound = net.bound - 1e-6
    pts = pts.clamp(-bound, bound)

    if dy_net != None:
        ray_phase = phase.reshape(n_rays, 1, 1)
        ray_phase = torch.tile(ray_phase, [1, n_samples, 1])
        comb = torch.cat([pts, ray_phase], dim=-1)


    if useAWP==True:
        raw ,deepfeature=run_network(comb, dy_net, netchunk, useAWP)
    else:
        raw =run_network(comb, dy_net, netchunk, useAWP)

    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std,pts)


    ret = {'acc': acc, 'pts':pts , 'raw':raw}

    if useAWP:
        ret['depth_feature'] = deepfeature
        ret['z_vals'] = z_vals
        ret['rays_d'] = rays_d

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



def run_network(inputs, fn, netchunk,useAWP=False):
    """
    Prepares inputs and applies network 'fn'.
    """
    if isinstance(inputs, torch.Tensor)!= True:
        inputs = torch.tensor(inputs,dtype=torch.float32,device='cuda')

    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = torch.cat([fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    if useAWP==True:
        deepth_feauture = torch.cat([fn(uvt_flat[i:i + netchunk])[1] for i in range(0, uvt_flat.shape[0], netchunk)], 0)
        feauture = deepth_feauture.reshape(list(inputs.shape[:-1]) + [-1])

        return out,feauture
    else:
        return out

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.,pts=0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    # dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, torch.Tensor([1e-10]).expand(dists[..., :1].shape).to(dists.device)], -1)
    # dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    sub = pts[:, 1:, :] - pts[:, :-1, :]
    d = torch.norm(sub, dim=-1)
    dists = torch.cat([d, torch.Tensor([1e-10]).expand(d[:, :1].shape).to(pts.device)], -1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std
        noise = noise.to(raw.device)

    acc = torch.sum((raw[..., 0] + noise) * dists, dim=-1)

    if raw.shape[-1] == 1:
        eps = torch.ones_like(raw[:, :1, -1]) * 1e-10
        weights = torch.cat([eps, torch.abs(raw[:, 1:, -1] - raw[:, :-1, -1])], dim=-1)
        weights = weights / torch.max(weights)
    elif raw.shape[-1] == 2: # with jac
        weights = raw[..., 1] / torch.max(raw[..., 1])
    else:
        raise NotImplementedError("Wrong raw shape")

    return acc, weights

def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


        
        





