# Fusion-TRFC-estimation-based-on-CA

## 项目简介
本项目实现了一种基于**交叉注意力机制（Cross-Attention）**的动态-视觉双模态特征级融合估计方法。该方法融合车辆动力学数据与视觉图像特征，用于提升轮胎-路面摩擦系数（TRFC）的估计精度与鲁棒性。

## 项目结构
项目数据集按照训练集与测试集划分，并包含两个核心运行脚本。

```text
Fusion-TRFC-estimation-based-on-CA/
│
├── README.md
├── train/                                  # 训练集目录
│   ├── episode_1__pi05_annotations_snow_clean/   # 示例训练序列
│   │   ├── dynamics_40Hz.csv               # 动力学数据
│   │   └── images/
│   │       └── ZED_LEFT/                   # 左目视觉图像序列
│   │           └── ZED_LEFT_xxx.jpg, ...   # 图像帧
│   └── ...                                 # 其他训练序列
│
├── test/                                   # 测试集目录
│   ├── episode_2__pi05_annotations_snow_clean/   # 示例测试序列
│   │   ├── dynamics_40Hz.csv
│   │   └── images/
│   │       └── ZED_LEFT/
│   │           └── ZED_LEFT_xxx.jpg, ...
│   └── ...
│
├── train_dyn_bias_drop_lr_40Hz.py          # 模型训练脚本
├── train_dyn_transformer.py                # 动力学预训练脚本
└── estimate-feature.py                     # 测试样本估计
```

使用说明  
运行train_dyn_trasnsformer.py 进行动力学预训练（可选）
运行 train_dyn_bias_drop_lr_40Hz.py 训练估计模型。  
运行 estimate-feature.py 对 test 文件夹中的样本进行估计。

数据引用
本项目使用了以下数据集：

极端路面图像数据集（Extreme-Road-Image-Dataset）
@article{zhao2025friction,
author={S. Zhao and J. Zhang and Y. Jiang and C. He and J. Han},
title={Tire-Road Friction Coefficients Adaptive Estimation through Image and Vehicle Dynamics Integration},
journal={Mechanical Systems and Signal Processing},
volume={224},
pages={112039},
year={2025},
doi={10.1016/j.ymssp.2024.112039}
}

极端驾驶条件数据集（Extreme_Driving_Conditions_Dataset）
@misc{extreme_driving_dataset_2026,
title={Extreme Driving Dataset: Multi-Modal Episodes for Critical and Adverse-Condition Driving},
author={Zhao, Shiyue and Li, Xinhan and Jiang, Yuhong and He, Chengkun and Zhang, Junzhi},
howpublished={Tsinghua University Intelligent Chassis Team},
year={2026}
}
获取地址：https://huggingface.co/datasets/Stary108/Extreme_Driving_Conditions_Dataset
