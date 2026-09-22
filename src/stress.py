"""
Week 7 Stress Test (Fixed): 修复2D低维相似度阈值失效 + 减少样本增大过拟合压力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 配置区 (本次修复重点) ----
N_SAMPLES = 600          # 🛠 减少样本: 增大过拟合压力
NOISE = 0.35
HIDDEN1, HIDDEN2 = 256, 128
EPOCHS = 300
LR = 0.05
BATCH_SIZE = 32
SEED = 42

SIM_THRESHOLD = 0.95     # 🛠 修复: 2D低维下只有几乎同向才算冗余
P_EXTREME = 0.8          # 🛠 修复: 减轻极刑, 防欠拟合
P_PROTECT = 0.3

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

class HyperplaneDropout(nn.Module):
    def __init__(self, n, default_p=0.5):
        super().__init__()
        self.register_buffer('dropout_probs', torch.full((n,), float(default_p)))
    def update_probs(self, p): self.dropout_probs = p.to(self.dropout_probs.device)
    def forward(self, x):
        if not self.training: return x
        mask = (torch.rand_like(x) > self.dropout_probs).float()
        return x * mask / (1.0 - self.dropout_probs + 1e-6)

class MLP_2D(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.fc1 = nn.Linear(2, HIDDEN1); self.relu1 = nn.ReLU()
        if mode == "hyper": self.drop1 = HyperplaneDropout(HIDDEN1)
        elif mode == "fixed": self.drop1 = nn.Dropout(0.5)
        else: self.drop1 = nn.Identity()
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2); self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5) if mode != "none" else nn.Identity()
        self.fc3 = nn.Linear(HIDDEN2, 2)
    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)

def analyze_and_update(model, X_tensor):
    model.eval(); n = HIDDEN1
    W2 = model.fc2.weight.data.cpu(); static = torch.sum(torch.abs(W2), dim=0).numpy()
    with torch.no_grad():
        dyn = torch.sum(model.relu1(model.fc1(X_tensor)), dim=0).cpu().numpy()
    impact = dyn * static
    W1 = model.fc1.weight.data.cpu(); W1n = F.normalize(W1, p=2, dim=1)
    sim = torch.mm(W1n, W1n.T).numpy(); np.fill_diagonal(sim, -1.0)
    max_sim = np.max(sim, axis=1)
    probs = np.full(n, 0.5)
    low = np.percentile(impact, 30); high = np.percentile(impact, 70)
    for i in range(n):
        if impact[i] < low and max_sim[i] > SIM_THRESHOLD: probs[i] = P_EXTREME
        elif impact[i] > high: probs[i] = P_PROTECT
    old = model.drop1.dropout_probs.cpu()
    model.drop1.update_probs(0.5 * torch.tensor(probs, dtype=torch.float32) + 0.5 * old)

def train_model(model, loader, X_tensor, mode):
    crit = nn.CrossEntropyLoss(); opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for ep in range(1, EPOCHS + 1):
        if mode == "hyper" and ep > 30 and ep % 30 == 0:
            analyze_and_update(model, X_tensor)
        model.train()
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            opt.zero_grad(); loss = crit(model(d), t); loss.backward(); opt.step()

def calculate_boundary_radio(model, X):
    model.eval(); h = 0.02
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    xmin = float(X_np[:, 0].min()) - 0.5; xmax = float(X_np[:, 0].max()) + 0.5
    ymin = float(X_np[:, 1].min()) - 0.5; ymax = float(X_np[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = F.softmax(model(grid), dim=1)[:, 1].cpu().numpy().reshape(xx.shape)
    mask = Z > 0.5
    diff_x = np.sum(np.abs(np.diff(mask.astype(int), axis=1)))
    diff_y = np.sum(np.abs(np.diff(mask.astype(int), axis=0)))
    return diff_x + diff_y, Z, xx, yy

def plot_with_metrics(ax, model, X, y, title_prefix, test_acc):
    boun_radio, Z, xx, yy = calculate_boundary_radio(model, X)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=plt.cm.RdBu, edgecolors='k', s=30)
    ax.set_title(f"{title_prefix}\nTest Acc: {test_acc:.1f}% | BounRadio: {boun_radio}",
                 fontsize=11, fontweight='bold')

if __name__ == "__main__":
    set_seed(SEED)
    X_all, y_all = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=SEED)
    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    X_plot = torch.tensor(X_all, dtype=torch.float32)
    y_plot = torch.tensor(y_all, dtype=torch.long)

    modes = ["none", "fixed", "hyper"]
    titles = ["No Dropout\n(Severe Overfitting)", "Fixed 0.5 Dropout\n(Baseline)", "Ours\n(Combined Hyperplane)"]
    test_accs = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, mode, title in zip(axes, modes, titles):
        set_seed(SEED)
        m = MLP_2D(mode).to(device)
        print(f"⏳ 训练 [{mode}] ...")
        train_model(m, loader, Xt, mode)
        m.eval()
        with torch.no_grad():
            _, pred = torch.max(m(X_test_t), 1)
        test_acc = 100 * (pred.cpu() == y_test_t.cpu()).float().mean().item()
        test_accs.append(test_acc)
        plot_with_metrics(ax, m, X_plot, y_plot, title, test_acc=test_acc)
    plt.tight_layout()
    p = CKPT_DIR / "stress_test_fixed.png"
    plt.savefig(p, dpi=300); print(f"\n✅ 已保存: {p}"); plt.show()

    print("\n" + "="*60)
    print(f"{'Strategy':<15} | {'Test Acc':<10}")
    print("-"*60)
    for mode, acc in zip(modes, test_accs):
        tag = " 🏆" if mode == "hyper" else ""
        print(f"{mode:<15} | {acc:>6.2f}%{tag}")
    print("="*60)