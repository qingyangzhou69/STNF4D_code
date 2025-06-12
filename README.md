# Spatiotemporal-aware Neural Fields for Dynamic CT Reconstruction


Spatiotemporal-aware Neural Fields for Dynamic CT Reconstruction [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/33177/35332)] [[Project Page](https://qingyangzhou69.github.io/STNF4D/)].

[Qingyang Zhou](),[Yunfan Ye](https://yunfan1202.github.io),  [Zhiping Cai]().

![](./figures/teaser.gif)

### Abstract
We propose a dynamic Computed Tomography (CT) reconstruction framework called STNF4D (SpatioTemporal-aware Neural Fields). 
First, we represent the 4D scene using four orthogonal volumes and compress these volumes into more compact hash grids. 
Compared to the plane decomposition method, this method enhances the model's capacity while keeping
the representation compact and efficient. However, in densely predicted high-resolution dynamic CT scenes,
the lack of constraints and hash conflicts in the hash grid features lead to obvious dot-like artifact and
blurring in the reconstructed images. To address these issues, we propose the Spatiotemporal Transformer
(ST-Former) that guides the model in selecting and optimizing features by sensing the spatiotemporal information
in different hash grids, significantly improving the quality of reconstructed images. We conducted experiments on
medical and industrial datasets covering various motion types, sampling modes, and reconstruction resolutions.
Experimental results show that our method outperforms the second-best by 5.99 dB and 4.27 dB in medical and industrial scenes, respectively.

![](./figures/pipeline.png)

### Installation

``` sh
# Create envorinment
conda create -n naf python=3.9
conda activate naf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```
After testing, higher versions of torch can also run :)
### Training

Download 4D CT datasets from [here](). Put them into the `./data` folder.

Experiments settings are stored in `./config` folder.

For example, train STNF4D with `XCAT` dataset:

``` sh
python train.py --config ./config/XCAT.yaml
```

### Testing and Reconstruction

The trained weights will be stored in `./log` folder.

For example, test with `XCAT` dataset:

``` sh
python reconstruction.py --config ./config/XCAT.yaml
```

# Acknowledgement
[torch-ngp](https://github.com/ashawkey/torch-ngp.git)

[NAF](https://github.com/Ruyi-Zha/naf_cbct.git)

[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF.git)

[TIGRE toolbox](https://github.com/CERN/TIGRE.git)