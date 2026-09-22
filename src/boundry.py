"""
Week 7: 2D 决策边界对比图 (Decision Boundary Visualization)
核心目的: 直观展示 Baseline 的过拟合毛刺 vs 我们的 Combined 方法的平滑几何边界
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
import time

# ============ 0. 基础设置 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2D 数据训练极快，我们多跑几个 Epoch 让边界充分拟合(包括过拟合的毛刺)
EPOCHS = 150
LR = 0.05
BATCH_SIZE = 64
SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============ 1. 复用我们的核心武器: HyperplaneDropout ============
class HyperplaneDropout(nn.Module):
    def __init__(self, num_features, default_p=0.5):
        super().__init__()
        self.register_buffer('dropout_probs', torch.full((num_features,), float(default_p)))

    def update_probs(self, new_probs):
        self.dropout_probs = new_probs.to(self.dropout_probs.device)

    def forward(self, x):
        if not self.training: return x
        mask = (torch.rand_like(x) > self.dropout_probs).float()
        scale = 1.0 / (1.0 - self.dropout_probs + 1e-6)
        return x * mask * scale


# ============ 2. 定义适配 2D 数据的 MLP ============
class MLP_2D(nn.Module):
    def __init__(self, use_hyperplane_dropout=False):
        super().__init__()
        self.use_hd = use_hyperplane_dropout
        # 第一层故意设大一点(64)，容易产生冗余超平面
        self.fc1 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU()
        self.drop1 = HyperplaneDropout(64) if use_hyperplane_dropout else nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)  # 第二层保持常规

        self.fc3 = nn.Linear(32, 2)  # 二分类输出

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)


# ============ 3. 多维超平面分析 (适配 2D MLP) ============
def analyze_and_update(model, X_train):
    model.eval()
    num_neurons = 64

    # 计算 Impact
    W2 = model.fc2.weight.data.cpu()
    static = torch.sum(torch.abs(W2), dim=0).numpy()
    with torch.no_grad():
        a1 = model.relu1(model.fc1(X_train))
        dyn = torch.sum(a1, dim=0).cpu().numpy()
    impact = dyn * static

    # 计算 Similarity
    W1 = model.fc1.weight.data.cpu()
    W1n = F.normalize(W1, p=2, dim=1)
    sim = torch.mm(W1n, W1n.T).numpy()
    np.fill_diagonal(sim, -1.0)
    max_sim = np.max(sim, axis=1)

    # 分配概率 (Combined 逻辑)
    probs = np.full(num_neurons, 0.5)
    low_thr = np.percentile(impact, 30)
    high_thr = np.percentile(impact, 70)

    for i in range(num_neurons):
        if impact[i] < low_thr and max_sim[i] > 0.7:  # 双重该死 -> 极刑
            probs[i] = 0.9
        elif impact[i] > high_thr:  # 核心 -> 保护
            probs[i] = 0.2

    # 平滑更新
    old = model.drop1.dropout_probs.cpu()
    new_p = torch.tensor(probs, dtype=torch.float32)
    model.drop1.update_probs(0.5 * new_p + 0.5 * old)


# ============ 4. 训练函数 ============
def train_model(model, train_loader, X_train_tensor, is_ours):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        # 动态更新
        if is_ours and epoch > 20 and epoch % 20 == 0:
            analyze_and_update(model, X_train_tensor)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()


# ============ 5. 核心画图函数 ============
def plot_decision_boundary(ax, model, X, y, title):
    model.eval()
    # 生成覆盖整个空间的密集网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 将网格点送入模型预测
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        # 使用 softmax 获取属于类别 1 的概率
        out = F.softmax(model(grid_tensor), dim=1)[:, 1].cpu().numpy()
    Z = out.reshape(xx.shape)

    # 画等高线背景
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    # 画真实的散点
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', s=40)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


# ============ 6. 主流程 ============
if __name__ == "__main__":
    set_seed(SEED)
    print("🚀 生成 2D 双月数据集 (make_moons)...")
    # 生成带有噪声的双月数据
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # --- 训练 Baseline ---
    print("⏳ 正在训练 Baseline (Fixed Dropout)...")
    model_base = MLP_2D(use_hyperplane_dropout=False).to(device)
    train_model(model_base, train_loader, X_train_tensor, is_ours=False)

    # --- 训练 Ours ---
    print("⏳ 正在训练 Ours (Combined Hyperplane Dropout)...")
    set_seed(SEED)  # 保证初始权重一致
    model_ours = MLP_2D(use_hyperplane_dropout=True).to(device)
    train_model(model_ours, train_loader, X_train_tensor, is_ours=True)

    # --- 画图对比 ---
    print("🎨 正在绘制惊艳的决策边界对比图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_decision_boundary(ax1, model_base, X, y,
                           "Baseline (Fixed 0.5 Dropout)\nNotice the fragmented boundaries & islands")
    plot_decision_boundary(ax2, model_ours, X, y,
                           "Ours (Combined Hyperplane Dropout)\nSmoother, more confident geometric boundaries")

    plt.tight_layout()
    save_path = CKPT_DIR / "decision_boundary_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 绝美对比图已保存至: {save_path}")
    plt.show()