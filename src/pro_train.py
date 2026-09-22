"""
Week 5: 超平面引导的自适应 Dropout (Hyperplane-Guided Adaptive Dropout)
整合: Week2 余弦相似度(冗余度) + Week4 Impact Score(贡献度)
特性: 静态分配 + 周期性动态更新 + 概率平滑 (防Loss震荡)
自包含文件: 不依赖 baseline.py / custom_layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import time

# ============ 0. 路径与设备 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 1. 超参数配置区 (所有调参都在这里) ============
BATCH_SIZE = 128          # MX250 安全值
EPOCHS = 10
LR = 0.001
SEED = 42

# --- 超平面分析超参数 ---
SIM_THRESHOLD = 0.7       # Week2 发现的黄金冗余阈值 (46对)
IMPACT_LOW_PCT = 30       # 贡献度后30% = 低贡献
IMPACT_HIGH_PCT = 70      # 贡献度前30% = 高贡献
P_EXTREME = 0.9           # 极刑概率 (低贡献 + 高冗余)
P_PROTECT = 0.2           # 保护概率 (高贡献)
P_DEFAULT = 0.5           # 默认概率

# --- 动态更新超参数 ---
WARMUP_EPOCHS = 2         # 前2个epoch用均匀0.5热身 (权重还随机, 分析无意义)
UPDATE_INTERVAL = 2       # 之后每2个epoch重新分析一次
SMOOTH_ALPHA = 0.5        # 概率平滑: 新 = α*新计算 + (1-α)*旧 (防震荡)
ANALYZE_BATCHES = 10      # 分析时抽样的batch数 (越大越准但越慢)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# ============ 2. 自定义超平面 Dropout 层 ============
class HyperplaneDropout(nn.Module):
    """允许为每个神经元分配不同 Dropout 概率的层"""
    def __init__(self, num_features, default_p=0.5):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('dropout_probs', torch.full((num_features,), float(default_p)))

    def update_probs(self, new_probs):
        self.dropout_probs = new_probs.to(self.dropout_probs.device)

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand_like(x) > self.dropout_probs).float()
        scale = 1.0 / (1.0 - self.dropout_probs + 1e-6)  # Inverted Dropout
        return x * mask * scale

# ============ 3. 模型定义 ============
class HyperplaneMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.hp_dropout1 = HyperplaneDropout(256)   # 第一层: 自适应
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.hp_dropout2 = HyperplaneDropout(128)   # 第二层: 暂用默认0.5
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.hp_dropout1(self.relu1(self.fc1(x)))
        x = self.hp_dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============ 4. 数据加载 ============
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
analyze_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)  # 分析专用(固定顺序)

# ============ 5. 多维超平面分析 ============
def analyze_and_get_probs(model, num_neurons=256):
    """融合 Impact(贡献度) + CosineSim(冗余度), 返回概率Tensor和统计"""
    model.eval()
    # 5.1 Impact Score
    W2 = model.fc2.weight.data.cpu()
    static_impact = torch.sum(torch.abs(W2), dim=0).numpy()
    dynamic = np.zeros(num_neurons)
    with torch.no_grad():
        for i, (data, _) in enumerate(analyze_loader):
            if i >= ANALYZE_BATCHES: break
            data = data.to(device).view(-1, 28*28)
            a1 = model.relu1(model.fc1(data))
            dynamic += torch.sum(a1, dim=0).cpu().numpy()
    impact = dynamic * static_impact

    # 5.2 余弦相似度 -> 每个神经元的最大相似度
    W1 = model.fc1.weight.data.cpu()
    W1n = F.normalize(W1, p=2, dim=1)
    sim = torch.mm(W1n, W1n.T).numpy()
    np.fill_diagonal(sim, -1.0)
    max_sim = np.max(sim, axis=1)

    # 5.3 融合分配概率
    low_thr = np.percentile(impact, IMPACT_LOW_PCT)
    high_thr = np.percentile(impact, IMPACT_HIGH_PCT)
    probs = np.full(num_neurons, P_DEFAULT)
    n_extreme = n_protect = 0
    for i in range(num_neurons):
        if impact[i] < low_thr and max_sim[i] > SIM_THRESHOLD:
            probs[i] = P_EXTREME; n_extreme += 1      # 双重该死 -> 极刑
        elif impact[i] > high_thr:
            probs[i] = P_PROTECT; n_protect += 1      # 核心 -> 保护
    stats = dict(extreme=n_extreme, protect=n_protect, default=num_neurons-n_extreme-n_protect)
    return torch.tensor(probs, dtype=torch.float32), stats

def apply_probs(model, new_probs):
    """带平滑地更新概率, 防止Loss震荡"""
    old = model.hp_dropout1.dropout_probs.cpu()
    smoothed = SMOOTH_ALPHA * new_probs + (1 - SMOOTH_ALPHA) * old
    model.hp_dropout1.update_probs(smoothed)

# ============ 6. 训练与评估 ============
model = HyperplaneMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate():
    model.eval(); correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            _, pred = torch.max(out, 1)
            total += target.size(0); correct += (pred == target).sum().item()
    return 100 * correct / total

def train():
    print("--- Week5: 超平面引导自适应 Dropout 训练 ---")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        # 动态更新: warmup后, 每隔 UPDATE_INTERVAL 个epoch重新分析
        if epoch > WARMUP_EPOCHS and (epoch - WARMUP_EPOCHS) % UPDATE_INTERVAL == 1:
            probs, stats = analyze_and_get_probs(model)
            apply_probs(model, probs)
            print(f"  [Epoch {epoch}] 超平面再分析 -> 极刑={stats['extreme']}, 保护={stats['protect']}, 默认={stats['default']}")
        model.train()
        running = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data); loss = criterion(out, target)
            loss.backward(); optimizer.step()
            running += loss.item()
        print(f"Epoch [{epoch:02d}/{EPOCHS}] Loss: {running/len(train_loader):.4f}")
    print(f"⏱️ 耗时: {time.time()-start:.1f}s")

# ============ 7. 主流程 ============
if __name__ == "__main__":
    train()
    acc = evaluate()
    print("=" * 40)
    print(f"🎯 Week5 超平面Dropout 准确率: {acc:.2f}%")
    print(f"📏 Baseline 对照: 97.80%")
    print("=" * 40)
    torch.save(model.state_dict(), CKPT_DIR / "week5_hyperplane.pth")
    print(f"💾 权重已保存: {CKPT_DIR / 'week5_hyperplane.pth'}")