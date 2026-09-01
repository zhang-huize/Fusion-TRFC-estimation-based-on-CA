#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅使用特征级融合模型进行估计（推理）
输入：每个 episode 文件夹下的 dynamics_40Hz.csv（包含特征列、rac、road_image_path）
输出：估计值数组，保存为 ./result/feature-fusion/episode_name.npy
配置参数请根据实际情况修改。
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm

# ==================== 配置（请根据训练时的设置修改）====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据根目录（包含各 episode 子文件夹）
train_root = Path("./test")

# 结果保存根目录
result_root = Path("./result")
output_subdir = "feature-fusion"       # 子目录名

# 模型及统计量路径
fusion_model_path = "fusion_CA_rac_seq4_img4_40Hz.pth"   # 与训练保存的名称一致
stats_path = "dyn_feat_stats_rac_40Hz.npz"               # 40Hz 数据对应的统计量

# 滑动窗口参数（必须与训练时完全相同）
seq_len = 4
frame_idx_for_image = seq_len - 1   # 取窗口最后一帧（索引3）

# 动力学特征列名（顺序必须与训练一致）
FEATURE_COLS = [
    'speed_kmh', 'ax_g', 'ay_g', 'yaw_rate_dps', 'steering_angle_deg',
    'wheel_fl_kmh', 'wheel_fr_kmh', 'wheel_rl_kmh', 'wheel_rr_kmh',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr'
]

# 图像预处理（与训练一致）
img_size = 224
img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ================================================================

# ==================== 模型定义（与训练时完全一致）====================
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_lite0', pretrained=False, num_classes=0)
        self.fc = nn.Linear(1280, out_dim)
    def forward(self, x):
        return self.fc(self.backbone(x))

class DynamicsEncoder(nn.Module):
    def __init__(self, input_dim=13, d_model=128, nhead=8, num_layers=4, out_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dyn_dim, img_dim, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.dyn_proj = nn.Linear(dyn_dim, embed_dim)
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.cross_attn_i2d = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_final = nn.LayerNorm(embed_dim)
    def forward(self, dyn_feat, img_feat):
        dyn = self.dyn_proj(dyn_feat).unsqueeze(0)
        img = self.img_proj(img_feat).unsqueeze(0)
        attn, _ = self.cross_attn_i2d(img, dyn, dyn)
        img_out = self.norm_img(img + attn).squeeze(0)
        dyn_out = dyn.squeeze(0)
        combined = torch.cat([dyn_out, img_out], dim=1)
        fused = self.ffn(combined)
        return self.norm_final(fused)

class RoadFrictionEstimator(nn.Module):
    def __init__(self, img_out_dim=32, dyn_out_dim=128, embed_dim=128):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=img_out_dim)
        self.dyn_encoder = DynamicsEncoder(out_dim=dyn_out_dim)
        self.fusion = CrossAttentionFusion(dyn_out_dim, img_out_dim, embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, dyn_feat, img):
        img_feat = self.img_encoder(img)
        dyn_feat = self.dyn_encoder(dyn_feat)
        fused = self.fusion(dyn_feat, img_feat)
        return self.regressor(fused)
# ================================================================

def load_episode_data(csv_path):
    """从 CSV 读取特征、真值和图像路径"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    N = len(rows)
    feats = np.zeros((N, len(FEATURE_COLS)), dtype=np.float32)
    rac_true = np.zeros(N, dtype=np.float32)
    img_paths = []
    for i, row in enumerate(rows):
        for j, col in enumerate(FEATURE_COLS):
            feats[i, j] = float(row[col])
        rac_true[i] = float(row['rac'])
        img_paths.append(row['road_image_path'])
    return feats, rac_true, img_paths

def compute_feature_fusion_estimation(model, feats, img_paths, means, stds):
    """
    对单个 episode 的所有有效帧（从 seq_len-1 到 N-1）进行特征级融合估计
    返回估计数组（长度为 N，前 seq_len-1 个为 nan）
    """
    N = feats.shape[0]
    est = np.full(N, np.nan, dtype=np.float32)
    for t in range(seq_len - 1, N):
        start = t - seq_len + 1
        end = t + 1
        # 窗口特征 (13, seq_len)
        window = feats[start:end, :].T
        # 归一化
        window_norm = (window - means[:, None]) / (stds[:, None] + 1e-8)
        dyn_tensor = torch.from_numpy(window_norm).unsqueeze(0).to(device)

        # 取窗口内第 frame_idx_for_image 帧的图像
        img_idx = start + frame_idx_for_image
        img_path = img_paths[img_idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            img = Image.open(Path.cwd() / img_path).convert('RGB')
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            mu = model(dyn_tensor, img_tensor).cpu().item()
        est[t] = np.clip(mu, 0.01, 1.0)
    return est

def main():
    # 加载模型
    print(f"使用设备: {device}")
    print("加载特征级融合模型...")
    model = RoadFrictionEstimator().to(device)
    if not os.path.exists(fusion_model_path):
        raise FileNotFoundError(f"模型文件不存在: {fusion_model_path}")
    model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    model.eval()

    # 加载统计量
    print("加载动力学统计量...")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"统计量文件不存在: {stats_path}")
    stats = np.load(stats_path)
    means, stds = stats['means'], stats['stds']

    # 创建输出目录
    out_dir = result_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 episode（包含 dynamics_40Hz.csv 的目录）
    episode_dirs = [d for d in train_root.iterdir() if d.is_dir() and (d / "dynamics_40Hz.csv").exists()]
    if not episode_dirs:
        print(f"警告: 在 {train_root} 下未找到任何包含 dynamics_40Hz.csv 的目录")
        return
    print(f"找到 {len(episode_dirs)} 个 episode")

    for ep_dir in episode_dirs:
        csv_path = ep_dir / "dynamics_40Hz.csv"
        print(f"\n处理: {ep_dir.name}")
        feats, rac_true, img_paths = load_episode_data(csv_path)
        N = feats.shape[0]
        if N < seq_len:
            print(f"  跳过: 帧数 {N} 小于窗口长度 {seq_len}")
            continue

        # 计算特征级融合估计
        est = compute_feature_fusion_estimation(model, feats, img_paths, means, stds)
        # 保存结果
        save_path = out_dir / f"{ep_dir.name}.npy"
        np.save(save_path, est)
        print(f"  估计已保存至 {save_path}，有效估计数: {np.sum(~np.isnan(est))}")

    print("\n所有估计完成。")

if __name__ == "__main__":
    main()