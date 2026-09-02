# Fusion-TRFC-estimation-based-on-CA

## Project Description
This project implements a dynamic-visual dual-modal feature-level fusion estimation method based on **Cross-Attention mechanism**. It fuses vehicle dynamics data and visual image features to improve the accuracy and robustness of tire-road friction coefficient (TRFC) estimation.

## Project Structure
The dataset is divided into training and test sets, with two core execution scripts.

```
Fusion-TRFC-estimation-based-on-CA/
│
├── README.md
├── train/                                  # Training set directory
│   ├── episode_1__pi05_annotations_snow_clean/   # Example training sequence
│   │   ├── dynamics_40Hz.csv               # Dynamics data
│   │   └── images/
│   │       └── ZED_LEFT/                   # Left camera image sequence
│   │           └── ZED_LEFT_xxx.jpg, ...   # Image frames
│   └── ...                                 # Other training sequences
│
├── test/                                   # Test set directory
│   ├── episode_2__pi05_annotations_snow_clean/   # Example test sequence
│   │   ├── dynamics_40Hz.csv
│   │   └── images/
│   │       └── ZED_LEFT/
│   │           └── ZED_LEFT_xxx.jpg, ...
│   └── ...
│
├── train_dyn_bias_drop_lr_40Hz.py          # Model training script
├── train_dyn_transformer.py                # Dynamics pre-training script
└── estimate-feature.py                     # Test sample estimation
```

Usage  
Run train_dyn_transformer.py to pre-train dynamics encoder.  
Run train_dyn_bias_drop_lr_40Hz.py to train the estimation model.  
Run estimate-feature.py to estimate samples in the test folder.

Data References
The following datasets are used in this project:

Extreme-Road-Image-Dataset
@article{zhao2025friction,
author={S. Zhao and J. Zhang and Y. Jiang and C. He and J. Han},
title={Tire-Road Friction Coefficients Adaptive Estimation through Image and Vehicle Dynamics Integration},
journal={Mechanical Systems and Signal Processing},
volume={224},
pages={112039},
year={2025},
doi={10.1016/j.ymssp.2024.112039}
}

Extreme_Driving_Conditions_Dataset
@misc{extreme_driving_dataset_2026,
title={Extreme Driving Dataset: Multi-Modal Episodes for Critical and Adverse-Condition Driving},
author={Zhao, Shiyue and Li, Xinhan and Jiang, Yuhong and He, Chengkun and Zhang, Junzhi},
howpublished={Tsinghua University Intelligent Chassis Team},
year={2026}
}
Available at: https://huggingface.co/datasets/Stary108/Extreme_Driving_Conditions_Dataset
