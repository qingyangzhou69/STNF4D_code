# Medical dataset

## Download raw data

We use data from open source 4D medical CT datasets [4D-Lung Cancer Imaging Archive (4D-LCIA)](https://www.cancerimagingarchive.net/collection/4d-lung/).  
Download raw data and put them into `./data_generator/medical_dataset` folder as:


```sh
└── data_generator   
│   └── medical_dataset
│   │   └── 4Dlung
│   │   │   ├── 100_HM
│   │   │   │   ├──  Phase1
│   │   │   │   │    ├── 1-001.dcm
│   │   │   │   │    ├── 1-002.dcm
│   │   │   │   │    └── ...
│   │   │   │   ├──  Phase2
│   │   │   │   │    ├── 1-001.dcm
│   │   │   │   │    ├── 1-002.dcm
│   │   │   │   │    └── ...
│   │   │   │   ├── ...
│   │   │   ├── 101_HM
│   │   │   │   ├──  Phase1
│   │   │   │   │    ├── 1-001.dcm
│   │   │   │   │    ├── 1-002.dcm
│   │   │   │   │    └── ...
│   │   │   │   ├──  Phase2
│   │   │   │   │    ├── 1-001.dcm
│   │   │   │   │    ├── 1-002.dcm
│   │   │   │   │    └── ...
│   │   │   │   ├── ...
│   │   │   └── ...

```

## Process data and generate .pickle files

Run [generate_med_data.py](generate_med_data.py) to process the raw data and generate `.pickle` files:

```sh
python generate_med_data.py 
```
