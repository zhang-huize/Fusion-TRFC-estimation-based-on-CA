# train_dyn_bias_drop_lr_rac_40Hz.py
# 特征级融合：Transformer + EfficientNet-Lite0 + Cross-Attention
# 真值 = rac，滑动窗口长度 = 4，图像取窗口内最后一帧（第4帧）
# 数据源：dynamics_40Hz.csv

import os
import csv
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 13
img_size = 224
img_out_dim = 32
dyn_out_dim = 128
embed_dim = dyn_out_dim

seq_len = 4                       # 修改：窗口长度 4
stride = 1
frame_idx_for_image = seq_len - 1 # 索引为 3（第4帧）

batch_size = 64
num_workers = min(6, os.cpu_count())
prefetch_factor = 2
persistent_workers = True if num_workers > 0 else False

fusion_epochs = 10
lr = 1e-4
r = 0.5
prob_drop_dyn = 0.01
prob_drop_img = 0.01
lambda_bias = 0.01

train_root = Path("./train")
pretrain_model_path = "dyn_transformer_rac_40Hz.pth"   # 使用新训练的动力学模型
stats_path = "dyn_feat_stats_rac_40Hz.npz"             # 新统计文件
model_save_path = f"fusion_CA_rac_seq{seq_len}_img{frame_idx_for_image+1}_40Hz.pth"

FEATURE_COLS = [
    'speed_kmh', 'ax_g', 'ay_g', 'yaw_rate_dps', 'steering_angle_deg',
    'wheel_fl_kmh', 'wheel_fr_kmh', 'wheel_rl_kmh', 'wheel_rr_kmh',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr'
]

train_ratio = 0.8

# ==================== 数据集（读取 dynamics_40Hz.csv）====================
class FusionDataset(Dataset):
    def __init__(self, samples, targets, img_paths, transform, means=None, stds=None):
        self.samples = [s.astype(np.float32) for s in samples]
        self.targets = np.array(targets, dtype=np.float32)
        self.img_paths = img_paths
        self.transform = transform
        self.means = means
        self.stds = stds
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        dyn_feat = self.samples[idx]
        if self.means is not None and self.stds is not None:
            dyn_feat = (dyn_feat - self.means[:, None]) / (self.stds[:, None] + 1e-8)
        dyn_tensor = torch.from_numpy(dyn_feat)
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            full_path = Path(img_path)
            if not full_path.is_absolute():
                full_path = Path.cwd() / img_path
            image = Image.open(full_path).convert('RGB')
        img_tensor = self.transform(image)
        label = torch.tensor([self.targets[idx]], dtype=torch.float32)
        return dyn_tensor, img_tensor, label, 0

def build_fusion_dataset(train_root, seq_len, stride, frame_idx, transform, means=None, stds=None):
    all_samples, all_targets, all_img_paths = [], [], []
    episode_dirs = [d for d in train_root.iterdir() if d.is_dir() and (d / "dynamics_40Hz.csv").exists()]
    print(f"找到 {len(episode_dirs)} 个 episode（40Hz 数据）")
    for ep_dir in episode_dirs:
        csv_path = ep_dir / "dynamics_40Hz.csv"
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            required = FEATURE_COLS + ['rac', 'road_image_path']
            if any(col not in reader.fieldnames for col in required):
                missing = [col for col in required if col not in reader.fieldnames]
                print(f"  跳过 {ep_dir.name}: 缺少 {missing}")
                continue
            feats, targets, img_paths = [], [], []
            for row in rows:
                feats.append([float(row[col]) for col in FEATURE_COLS])
                targets.append(float(row['rac']))
                img_paths.append(row['road_image_path'])
            feats = np.array(feats, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)
            img_paths = np.array(img_paths)
            num_frames = feats.shape[0]
            max_start = num_frames - seq_len + 1
            if max_start <= 0:
                print(f"  {ep_dir.name}: 帧数不足 {num_frames} < {seq_len}")
                continue
            for start in range(0, max_start, stride):
                end = start + seq_len
                window_feat = feats[start:end, :].T   # (13, seq_len)
                all_samples.append(window_feat)
                all_targets.append(targets[end-1])
                all_img_paths.append(img_paths[start + frame_idx])
            print(f"  {ep_dir.name}: {max_start} 个样本")
        except Exception as e:
            print(f"  跳过 {ep_dir.name}: {e}")
    if not all_samples:
        raise RuntimeError("无有效样本")
    return FusionDataset(all_samples, all_targets, all_img_paths, transform, means, stds)

def load_dyn_stats(stats_path):
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return stats['means'].astype(np.float32), stats['stds'].astype(np.float32)
    else:
        print("警告: 未找到统计量文件")
        return None, None

# ==================== 模型定义（不变）====================
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0)
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
    def __init__(self, dyn_dim, img_dim, embed_dim, num_heads=1, dropout=0.1, img_dropout=0.2):
        super().__init__()
        self.dyn_proj = nn.Linear(dyn_dim, embed_dim)
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.cross_attn_i2d = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.img_dropout = nn.Dropout(img_dropout)
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
        img_out = self.img_dropout(img_out)
        dyn_out = dyn.squeeze(0)
        combined = torch.cat([dyn_out, img_out], dim=1)
        fused = self.ffn(combined)
        return self.norm_final(fused)

class RoadFrictionEstimator(nn.Module):
    def __init__(self, img_out_dim, dyn_out_dim, embed_dim):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=img_out_dim)
        self.dyn_encoder = DynamicsEncoder(input_dim=feature_dim, out_dim=dyn_out_dim)
        self.fusion = CrossAttentionFusion(dyn_dim=dyn_out_dim, img_dim=img_out_dim, embed_dim=embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, dyn_feat, img):
        img_feat = self.img_encoder(img)
        dyn_feat = self.dyn_encoder(dyn_feat)
        fused = self.fusion(dyn_feat, img_feat)
        return self.regressor(fused)

# ==================== 训练函数（不变）====================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    prob_drop_dyn, prob_drop_img, lambda_bias):
    model.train()
    total_loss = total_mse = total_bias = 0.0
    for dyn_feat, img, label, _ in loader:
        dyn_feat, img, label = dyn_feat.to(device), img.to(device), label.to(device)
        if prob_drop_dyn > 0:
            mask = torch.rand(dyn_feat.size(0), device=device) < prob_drop_dyn
            dyn_feat[mask] = 0.0
        if prob_drop_img > 0:
            mask = torch.rand(img.size(0), device=device) < prob_drop_img
            img[mask] = 0.0
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(dyn_feat, img)
            loss_mse = criterion(output, label)
            unique_vals = torch.unique(label)
            bias_sq = []
            for v in unique_vals:
                mask = (label == v)
                if mask.sum().item() >= 2:
                    pred_c = output[mask].squeeze()
                    bias_sq.append((pred_c.mean() - v).pow(2))
            bias_penalty = torch.stack(bias_sq).mean() if bias_sq else torch.tensor(0.0, device=device)
            loss = loss_mse + lambda_bias * bias_penalty
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = dyn_feat.size(0)
        total_loss += loss.item() * bs
        total_mse += loss_mse.item() * bs
        total_bias += bias_penalty.item() * bs
    n = len(loader.dataset)
    return total_loss/n, total_mse/n, total_bias/n

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for dyn_feat, img, label, _ in loader:
            dyn_feat, img, label = dyn_feat.to(device), img.to(device), label.to(device)
            out = model(dyn_feat, img)
            total_loss += criterion(out, label).item() * dyn_feat.size(0)
    return total_loss / len(loader.dataset)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==================== 主程序 ====================
if __name__ == '__main__':
    dyn_means, dyn_stds = load_dyn_stats(stats_path)
    full_dataset = build_fusion_dataset(train_root, seq_len, stride, frame_idx_for_image, transform,
                                        means=dyn_means, stds=dyn_stds)
    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    model = RoadFrictionEstimator(img_out_dim, dyn_out_dim, embed_dim).to(device)
    if os.path.exists(pretrain_model_path):
        state = torch.load(pretrain_model_path, map_location=device)
        dyn_state = {k.replace('dyn_encoder.', ''): v for k, v in state.items() if k.startswith('dyn_encoder.')}
        model.dyn_encoder.load_state_dict(dyn_state, strict=False)
        print("加载动力学预训练权重（40Hz 版本）")
    else:
        print("警告: 未找到动力学预训练模型")

    optimizer = optim.Adam([
        {'params': model.dyn_encoder.parameters(), 'lr': lr * r},
        {'params': model.img_encoder.parameters(), 'lr': lr},
        {'params': model.fusion.parameters(), 'lr': lr},
        {'params': model.regressor.parameters(), 'lr': lr}
    ], lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    train_losses, val_losses = [], []
    for epoch in range(1, fusion_epochs+1):
        train_loss, train_mse, train_bias = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            prob_drop_dyn, prob_drop_img, lambda_bias
        )
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss); val_losses.append(val_loss)
        print(f"Epoch {epoch:2d}/{fusion_epochs} | Train Loss: {train_loss:.6f} (MSE:{train_mse:.6f} Bias:{train_bias:.6f}) | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"模型保存至 {model_save_path}")

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Fusion Training (40Hz, seq=4)')
    plt.legend(); plt.grid()
    plt.savefig('fusion_rac_40Hz_training_curve.png')
    plt.close()