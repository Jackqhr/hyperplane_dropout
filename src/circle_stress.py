"""
Week 7 Ultimate: 同心圆极限施压 (Circles Stress Test)
目的: 使用闭合边界数据集，逼迫 Baseline 产生大量“孤岛”，凸显 Ours 的修剪能力
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 极限施压参数 ----
N_SAMPLES = 200  # 极少数据
NOISE = 0.25  # 适度噪声
FACTOR = 0.4  # 内外圆比例
HIDDEN1, HIDDEN2 = 256, 128  # 大模型
EPOCHS = 500  # 超长训练，逼迫过拟合
LR = 0.1  # 大学习率，增加震荡
BATCH_SIZE = 32
SEED = 42


def set_seed(s):
    torch.manual_seed(s);
    np.random.seed(s)


class HyperplaneDropout(nn.Module):
    def __init__(self, n, default_p=0.5):
        super().__init__();
        self.register_buffer('dropout_probs', torch.full((n,), float(default_p)))

    def update_probs(self, p): self.dropout_probs = p.to(self.dropout_probs.device)

    def forward(self, x):
        if not self.training: return x
        mask = (torch.rand_like(x) > self.dropout_probs).float()
        return x * mask / (1.0 - self.dropout_probs + 1e-6)


class MLP_2D(nn.Module):
    def __init__(self, mode):
        super().__init__();
        self.mode = mode
        self.fc1 = nn.Linear(2, HIDDEN1);
        self.relu1 = nn.ReLU()
        self.drop1 = HyperplaneDropout(HIDDEN1) if mode == "hyper" else (
            nn.Dropout(0.5) if mode == "fixed" else nn.Identity())
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2);
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5) if mode != "none" else nn.Identity()
        self.fc3 = nn.Linear(HIDDEN2, 2)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)


def analyze_and_update(model, X):
    model.eval();
    n = HIDDEN1
    W2 = model.fc2.weight.data.cpu();
    static = torch.sum(torch.abs(W2), dim=0).numpy()
    with torch.no_grad():
        dyn = torch.sum(model.relu1(model.fc1(X)), dim=0).cpu().numpy()
    impact = dyn * static
    W1 = model.fc1.weight.data.cpu();
    W1n = F.normalize(W1, p=2, dim=1)
    sim = torch.mm(W1n, W1n.T).numpy();
    np.fill_diagonal(sim, -1.0);
    max_sim = np.max(sim, axis=1)
    probs = np.full(n, 0.5);
    low = np.percentile(impact, 30);
    high = np.percentile(impact, 70)
    for i in range(n):
        if impact[i] < low and max_sim[i] > 0.7:
            probs[i] = 0.9
        elif impact[i] > high:
            probs[i] = 0.2
    old = model.drop1.dropout_probs.cpu()
    model.drop1.update_probs(0.5 * torch.tensor(probs, dtype=torch.float32) + 0.5 * old)


def train_model(model, loader, X, mode):
    crit = nn.CrossEntropyLoss();
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for ep in range(1, EPOCHS + 1):
        if mode == "hyper" and ep > 50 and ep % 50 == 0:
            analyze_and_update(model, X)
        model.train()
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            opt.zero_grad();
            loss = crit(model(d), t);
            loss.backward();
            opt.step()


def plot_boundary(ax, model, X, y, title):
    model.eval();
    h = 0.01  # 网格更密，细节更清晰
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    g = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = F.softmax(model(g), dim=1)[:, 1].cpu().numpy().reshape(xx.shape)
    # 使用更鲜明的配色
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, levels=50)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]);
    ax.set_yticks([])  # 隐藏坐标轴，更美观


if __name__ == "__main__":
    set_seed(SEED)
    print("🚀 生成同心圆数据集 (make_circles)...")
    X, y = make_circles(n_samples=N_SAMPLES, noise=NOISE, factor=FACTOR, random_state=SEED)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    modes = ["none", "fixed", "hyper"]
    titles = ["No Dropout\n(Severe Overfitting)", "Fixed 0.5 Dropout\n(Baseline)", "Ours\n(Combined Hyperplane)"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, mode, title in zip(axes, modes, titles):
        set_seed(SEED)
        m = MLP_2D(mode).to(device)
        print(f"⏳ 正在极限施压训练 [{mode}] (500 Epochs)...")
        train_model(m, loader, Xt, mode)
        plot_boundary(ax, m, X, y, title)

    plt.tight_layout()
    p = CKPT_DIR / "circles_stress_test.png"
    plt.savefig(p, dpi=300);
    print(f"✅ 已保存: {p}");
    plt.show()