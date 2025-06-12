# Industrial dataset

## Download raw data

We use data from open source 4D industrial CT datasets [LLNL D4DCT](https://library.ucsd.edu/dc/object/bb74156780).  
Download raw data and put them into `./data_generator/industrial_datasets` folder as:


```sh
└── data_generator   
│   └── industrial_datasets
│   │   └── 2563^LLNL
│   │   │   ├── D4DCT_DFM
│   │   │   │   ├──  S01_004
│   │   │   │   │    ├── 00_256_180_180_gt_f000.npy
│   │   │   │   │    ├── 00_256_180_180_gt_f001.npy
│   │   │   │   │    └── ...
│   │   │   │   ├──  s02_005
│   │   │   │   │    ├── 00_256_180_180_gt_f000.npy
│   │   │   │   │    ├── 00_256_180_180_gt_f001.npy
│   │   │   │   │    └── ...
│   │   │   │   ├──  S04_009
│   │   │   │   │    ├── 00_256_180_180_gt_f000.npy
│   │   │   │   │    ├── 00_256_180_180_gt_f001.npy
│   │   │   │   │    └── ...
│   │   │   │   ├── ...

```

## Process data and generate .pickle files

Run [generate_ind_data.py](generate_ind_data.py) to process the raw data and generate `.pickle` files:

```sh
python generate_ind_data.py 
```
