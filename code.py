import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset, Subset


# -------------------------------
# Utility Functions
# -------------------------------

def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute Structural Similarity Index (SSIM) between super‑resolved (sr) and high‑resolution (hr) images.

    Args:
        sr: Super‑resolved image tensor of shape (C, H, W) in [0, 1].
        hr: Ground‑truth high‑resolution image tensor of shape (C, H, W) in [0, 1].

    Returns:
        SSIM value as a float in [‑1, 1].
    """
    sr_np = sr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    hr_np = hr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(sr_np, hr_np, data_range=1.0, channel_axis=2)


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute Peak Signal‑to‑Noise Ratio (PSNR) in dB."""
    mse = torch.mean((sr - hr) ** 2)
    eps = 1e-10  # avoid log(0)
    psnr = 10 * torch.log10(1.0 / (mse + eps))
    return psnr.item()


# -------------------------------
# Dataset
# -------------------------------

class PairedImageDataset(Dataset):
    """Load paired LR / HR images from Urban100 (2× split)."""

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        lr_transform: transforms.Compose | None = None,
        hr_transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted(
            [f for f in os.listdir(lr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        self.hr_files = sorted(
            [f for f in os.listdir(hr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        assert (
            len(self.lr_files) == len(self.hr_files)
        ), "Mismatch between the number of LR and HR images."
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_tensor = self.lr_transform(lr_img) if self.lr_transform else transforms.ToTensor()(lr_img)
        hr_tensor = self.hr_transform(hr_img) if self.hr_transform else transforms.ToTensor()(hr_img)
        return lr_tensor, hr_tensor


# -------------------------------
# CNN Baseline (SRCNN‑like)
# -------------------------------

class CNN_SR(nn.Module):
    """Simple convolutional network for 2× super‑resolution."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (scale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


# -------------------------------
# Lightweight Vision Transformer
# -------------------------------

class Transformer_SR(nn.Module):
    """Compact ViT‑style model for 2× super‑resolution."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        scale_factor: int = 2,
        dropout: float = 0.1,
        max_hw: int = 256,  # maximum height/width supported
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.embedding = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)

        # learnable 2‑D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_hw * max_hw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruction = nn.Conv2d(embed_dim, 3 * (scale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.max_hw = max_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.embedding(x)  # (B, C, H, W)
        b, c, h, w = feat.shape
        assert h <= self.max_hw and w <= self.max_hw, f"Input size too large ({h}×{w})."

        # flatten to sequence (B, H*W, C)
        seq = feat.flatten(2).transpose(1, 2)
        pos = self.pos_embed[:, : h * w, :]
        seq = seq + pos

        seq = self.transformer_encoder(seq)
        seq = self.norm(seq) + seq  # residual LN
        feat = seq.transpose(1, 2).reshape(b, c, h, w)

        out = self.reconstruction(feat)
        out = self.pixel_shuffle(out)
        return out


# -------------------------------
# Tiny SwinIR (simplified)
# -------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class SwinIR(nn.Module):
    """Very small SwinIR‑style network for 2× super‑resolution."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 64,
        num_blocks: int = 6,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.shallow_feat = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(embed_dim) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (scale_factor**2), 3, 1, 1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(embed_dim, in_chans, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallow_feat(x)
        res = self.body(x)
        x = self.conv(res) + x  # residual connection
        x = self.upsample(x)
        return x


# -------------------------------
# Training / Validation Routines
# -------------------------------

def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_psnr = total_ssim = 0.0
    count = 0
    start = time.time()
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            with torch.amp.autocast(device_type="cuda"):
                sr = model(lr)
            total_psnr += compute_psnr(sr, hr)
            total_ssim += compute_ssim(sr, hr)
            count += 1
    elapsed = time.time() - start
    return total_psnr / count, total_ssim / count, elapsed


def validate_bilinear(loader: DataLoader, device: torch.device):
    total_psnr = total_ssim = 0.0
    count = 0
    start = time.time()
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            _, h, w = hr.shape[1:]
            sr = F.interpolate(lr, size=(h, w), mode="bilinear", align_corners=False)
            total_psnr += compute_psnr(sr, hr)
            total_ssim += compute_ssim(sr, hr)
            count += 1
    elapsed = time.time() - start
    return total_psnr / count, total_ssim / count, elapsed


# -------------------------------
# Main Entry
# -------------------------------

def main() -> None:
    # Paths to the Urban100 (×2) LR / HR folders – update as needed
    lr_dir = "./code/data/LOW"
    hr_dir = "./code/data/HIGH"

    # Transforms – LR images → 256×128, HR images → 512×256 (2× upscaling)
    lr_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
    ])
    hr_transform = transforms.Compose([
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
    ])

    # Load full dataset (assumed 100 images)
    full_dataset = PairedImageDataset(lr_dir, hr_dir, lr_transform, hr_transform)
    train_dataset = Subset(full_dataset, list(range(90)))  # first 90 → training
    val_dataset = Subset(full_dataset, list(range(90, 100)))  # last 10 → validation

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate models
    cnn_model = CNN_SR().to(device)
    trans_model = Transformer_SR(embed_dim=64, num_heads=4, num_layers=2).to(device)
    swin_model = SwinIR(embed_dim=64, num_blocks=6).to(device)

    # Optimizers & loss
    criterion = nn.L1Loss()
    opt_cnn = optim.Adam(cnn_model.parameters(), lr=1e-4)
    opt_trans = optim.Adam(trans_model.parameters(), lr=1e-4)
    opt_swin = optim.Adam(swin_model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler()
    epochs = 200

    # Statistics containers
    cnn_psnr_list, trans_psnr_list, swin_psnr_list = [], [], []
    cnn_ssim_list, trans_ssim_list, swin_ssim_list = [], [], []
    cnn_time = trans_time = swin_time = 0.0

    # ---------------------
    # Train CNN
    # ---------------------
    print("\n=== Training CNN ===")
    for epoch in range(epochs):
        cnn_model.train()
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = cnn_model(lr)
            loss = criterion(sr, hr)
            opt_cnn.zero_grad()
            loss.backward()
            opt_cnn.step()
        psnr, ssim_val, t = validate(cnn_model, val_loader, device)
        cnn_psnr_list.append(psnr)
        cnn_ssim_list.append(ssim_val)
        cnn_time += t
        print(f"Epoch {epoch + 1}/{epochs} – CNN: PSNR {psnr:.2f} dB | SSIM {ssim_val:.4f} | val time {t:.2f}s")

    # ---------------------
    # Train SwinIR
    # ---------------------
    print("\n=== Training SwinIR ===")
    for epoch in range(epochs):
        swin_model.train()
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            opt_swin.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                sr = swin_model(lr)
                loss = criterion(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(opt_swin)
            scaler.update()
        psnr, ssim_val, t = validate(swin_model, val_loader, device)
        swin_psnr_list.append(psnr)
        swin_ssim_list.append(ssim_val)
        swin_time += t
        print(f"Epoch {epoch + 1}/{epochs} – SwinIR: PSNR {psnr:.2f} dB | SSIM {ssim_val:.4f} | val time {t:.2f}s")

    # ---------------------
    # Train Transformer
    # ---------------------
    print("\n=== Training Transformer ===")
    for epoch in range(epochs):
        trans_model.train()
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            opt_trans.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                sr = trans_model(lr)
                loss = criterion(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(opt_trans)
            scaler.update()
        psnr, ssim_val, t = validate(trans_model, val_loader, device)
        trans_psnr_list.append(psnr)
        trans_ssim_list.append(ssim_val)
        trans_time += t
        print(f"Epoch {epoch + 1}/{epochs} – Transformer: PSNR {psnr:.2f} dB | SSIM {ssim_val:.4f} | val time {t:.2f}s")

    # ---------------------
    # Bilinear baseline
    # ---------------------
    print("\n=== Validating Bilinear Interpolation ===")
    bilinear_psnr, bilinear_ssim, bilinear_time = validate_bilinear(val_loader, device)
    print(f"Bilinear: PSNR {bilinear_psnr:.2f} dB | SSIM {bilinear_ssim:.4f} | val time {bilinear_time:.2f}s")

    # ---------------------
    # Save models
    # ---------------------
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(cnn_model.state_dict(), os.path.join(save_dir, "cnn_model.pth"))
    torch.save(trans_model.state_dict(), os.path.join(save_dir, "transformer_model.pth"))
    torch.save(swin_model.state_dict(), os.path.join(save_dir, "swin_model.pth"))
    print(f"Models saved to {save_dir}")

    # ---------------------
    # Visualization
    # ---------------------
    plt.figure(figsize=(8, 6))
    plt.plot(cnn_psnr_list, label="CNN")
    plt.plot(trans_psnr_list, label="Transformer")
    plt.plot(swin_psnr_list, label="SwinIR")
    plt.axhline(y=bilinear_psnr, label="Bilinear", linestyle="--", color="black")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(cnn_ssim_list, label="CNN")
    plt.plot(trans_ssim_list, label="Transformer")
    plt.plot(swin_ssim_list, label="SwinIR")
    plt.axhline(y=bilinear_ssim, label="Bilinear", linestyle="--", color="black")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    methods = ["CNN", "Transformer", "SwinIR"]
    times = [cnn_time, trans_time, swin_time]
    plt.bar(methods, times)
    plt.ylabel("Total Validation Time (s)")
    plt.title("Processing Time Comparison")
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
    main()
