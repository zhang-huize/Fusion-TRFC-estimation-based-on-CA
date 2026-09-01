# train_dyn_transformer_rac.py
# 动力学预训练：直接从 dynamics.csv 读取 13 维时序特征，真值 = rac（真实附着系数）
# 使用滑动窗口构造样本，Transformer 回归

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 13
dyn_out_dim = 128
d_model = 128
nhead = 8
num_layers = 4
dropout = 0.1

seq_len = 10                     # 滑动窗口长度（帧数）
stride = 1                       # 滑动步长
batch_size = 64
train_ratio = 0.8
pretrain_epochs = 20
lr = 1e-4

num_workers = min(4, os.cpu_count())
persistent_workers = num_workers > 0
prefetch_factor = 2 if num_workers > 0 else None

# 路径
train_root = Path("./train")
model_save_path = "dyn_transformer_rac.pth"
stats_save_path = "dyn_feat_stats_rac.npz"

# CSV 特征列名
FEATURE_COLS = [
    'speed_kmh', 'ax_g', 'ay_g', 'yaw_rate_dps', 'steering_angle_deg',
    'wheel_fl_kmh', 'wheel_fr_kmh', 'wheel_rl_kmh', 'wheel_rr_kmh',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr'
]

# ==================== 数据加载 ====================
def load_episode_data(csv_path, seq_len, stride):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if 'rac' not in reader.fieldnames:
        raise RuntimeError(f"{csv_path} 缺少列 'rac'")

    feats = []
    targets = []
    for row in rows:
        feat = [float(row[col]) for col in FEATURE_COLS]
        feats.append(feat)
        targets.append(float(row['rac']))   # 真值使用真实 rac

    feats = np.array(feats, dtype=np.float32)   # (N, 13)
    targets = np.array(targets, dtype=np.float32)

    num_frames = feats.shape[0]
    samples, labels = [], []
    for start in range(0, num_frames - seq_len + 1, stride):
        end = start + seq_len
        window_feat = feats[start:end, :].T    # (13, seq_len)
        samples.append(window_feat)
        labels.append(targets[end-1])          # 窗口最后一帧的真值
    return samples, labels

def build_dataset(train_root, seq_len, stride):
    all_samples, all_labels = [], []
    episode_dirs = [d for d in train_root.iterdir() if d.is_dir() and (d / "dynamics.csv").exists()]
    if not episode_dirs:
        raise FileNotFoundError(f"在 {train_root} 下未找到任何 dynamics.csv")
    print(f"找到 {len(episode_dirs)} 个 episode")
    for ep_dir in episode_dirs:
        csv_path = ep_dir / "dynamics.csv"
        try:
            samples, labels = load_episode_data(csv_path, seq_len, stride)
            all_samples.extend(samples)
            all_labels.extend(labels)
            print(f"  {ep_dir.name}: {len(samples)} 个样本")
        except Exception as e:
            print(f"  跳过 {ep_dir.name}: {e}")
    if not all_samples:
        raise RuntimeError("未成功加载任何样本")
    return np.array(all_samples), np.array(all_labels)

class DynamicsDataset(Dataset):
    def __init__(self, samples, labels, means=None, stds=None):
        self.samples = samples.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.means = means
        self.stds = stds
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.means is not None and self.stds is not None:
            x = (x - self.means[:, None]) / (self.stds[:, None] + 1e-8)
        return torch.from_numpy(x), torch.from_numpy(np.array([self.labels[idx]]))

def compute_stats(samples):
    N, C, T = samples.shape
    flat = samples.transpose(0,2,1).reshape(-1, C)
    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)
    stds[stds < 1e-6] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)

# ==================== 模型定义 ====================
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

class DynamicsOnlyEstimator(nn.Module):
    def __init__(self, dyn_out_dim):
        super().__init__()
        self.dyn_encoder = DynamicsEncoder(input_dim=feature_dim, out_dim=dyn_out_dim,
                                           d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(dyn_out_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, dyn_feat):
        return self.regressor(self.dyn_encoder(dyn_feat))

# ==================== 训练 ====================
def main():
    print("构建数据集...")
    samples, labels = build_dataset(train_root, seq_len, stride)
    print(f"总样本数: {len(samples)}")
    if os.path.exists(stats_save_path):
        stats = np.load(stats_save_path)
        means, stds = stats['means'], stats['stds']
        print("加载已有统计量")
    else:
        means, stds = compute_stats(samples)
        np.savez(stats_save_path, means=means, stds=stds)
        print("计算并保存统计量")

    total = len(samples)
    indices = np.random.permutation(total)
    split = int(train_ratio * total)
    train_dataset = DynamicsDataset(samples[indices[:split]], labels[indices[:split]], means, stds)
    val_dataset = DynamicsDataset(samples[indices[split:]], labels[indices[split:]], means, stds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    model = DynamicsOnlyEstimator(dyn_out_dim=dyn_out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    train_losses, val_losses = [], []
    for epoch in range(1, pretrain_epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:2d}/{pretrain_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"模型保存至 {model_save_path}")

    # 绘制曲线
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Dynamics Pretraining (True rac)')
    plt.legend()
    plt.grid()
    plt.savefig('dyn_rac_training_curve.png')
    plt.close()

if __name__ == "__main__":
    main()