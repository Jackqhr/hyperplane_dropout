"""
Week 6 Final: 绝对公平的多种子消融实验 (Ablation Study)
核心设计:
1. 所有策略(含Baseline)共享同一训练管线与动态更新流程;
   Baseline 的 analyze 返回全 0.5, 平滑后仍为 0.5, 代码路径完全一致。
2. 多种子跑 Mean±Std, 拒绝 cherry-picking。
3. 四策略对比: Baseline / Impact-Only / Sim-Only / Combined(Ours)。
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 1. 超参数配置区 ============
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
SEEDS = [42, 43, 44]        # 固定种子组, 所有策略共用

P_EXTREME = 0.8             # 极刑概率 (低贡献+高冗余)
P_PROTECT = 0.4             # 保护概率; 设 None 取消保护
SIM_THRESHOLD = 0.7         # 余弦相似度冗余阈值
IMPACT_LOW_PCT = 30
IMPACT_HIGH_PCT = 70

WARMUP = 2                  # 热身epoch数
UPDATE_INTERVAL = 2         # 动态更新间隔
SMOOTH_ALPHA = 0.5          # 概率平滑系数
ANALYZE_BATCHES = 10        # 分析抽样batch数

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ 2. 自定义层与模型 ============
class HyperplaneDropout(nn.Module):
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
        scale = 1.0 / (1.0 - self.dropout_probs + 1e-6)
        return x * mask * scale

class HyperplaneMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256); self.relu1 = nn.ReLU()
        self.hp_dropout1 = HyperplaneDropout(256)
        self.fc2 = nn.Linear(256, 128); self.relu2 = nn.ReLU()
        self.hp_dropout2 = HyperplaneDropout(128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.hp_dropout1(self.relu1(self.fc1(x)))
        x = self.hp_dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

# ============ 3. 数据加载 ============
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
analyze_loader = DataLoader(train_ds, batch_size=512, shuffle=False)

# ============ 4. 超平面分析 (按策略分配概率) ============
def analyze(model, strategy, num_neurons=256):
    """返回该策略下每个神经元的 Dropout 概率 Tensor"""
    model.eval()
    probs = np.full(num_neurons, 0.5)

    if strategy == "Baseline":
        # 绝对公平: 恒为0.5, 平滑后仍0.5, 但依然走更新流程
        return torch.tensor(probs, dtype=torch.float32)

    # --- Impact Score (贡献度) ---
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
    low_thr = np.percentile(impact, IMPACT_LOW_PCT)
    high_thr = np.percentile(impact, IMPACT_HIGH_PCT)

    # --- 余弦相似度 (冗余度) ---
    W1 = model.fc1.weight.data.cpu()
    W1n = F.normalize(W1, p=2, dim=1)
    sim = torch.mm(W1n, W1n.T).numpy()
    np.fill_diagonal(sim, -1.0)
    max_sim = np.max(sim, axis=1)

    # --- 按策略分配 ---
    if strategy == "Impact-Only":
        probs[impact < low_thr] = P_EXTREME
    elif strategy == "Sim-Only":
        probs[max_sim > SIM_THRESHOLD] = P_EXTREME
    elif strategy == "Combined":
        for i in range(num_neurons):
            if impact[i] < low_thr and max_sim[i] > SIM_THRESHOLD:
                probs[i] = P_EXTREME
            elif P_PROTECT is not None and impact[i] > high_thr:
                probs[i] = P_PROTECT

    return torch.tensor(probs, dtype=torch.float32)

# ============ 5. 单次训练+评估 ============
def run_one(strategy, seed):
    set_seed(seed)
    model = HyperplaneMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        # 【绝对公平】所有策略(含Baseline)都执行动态更新+平滑
        if epoch > WARMUP and (epoch - WARMUP) % UPDATE_INTERVAL == 1:
            new_probs = analyze(model, strategy)
            old = model.hp_dropout1.dropout_probs.cpu()
            smoothed = SMOOTH_ALPHA * new_probs + (1 - SMOOTH_ALPHA) * old
            model.hp_dropout1.update_probs(smoothed)

        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    model.eval(); correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, pred = torch.max(model(data), 1)
            total += target.size(0); correct += (pred == target).sum().item()
    return 100 * correct / total

# ============ 6. 主流程: 多策略 × 多种子 ============
if __name__ == "__main__":
    strategies = ["Baseline", "Impact-Only", "Sim-Only", "Combined"]
    results = {}
    total_start = time.time()

    for strat in strategies:
        accs = []
        for seed in SEEDS:
            acc = run_one(strat, seed)
            accs.append(acc)
            print(f"  [{strat}] seed={seed} -> {acc:.2f}%")
        results[strat] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"✅ [{strat}] Mean={results[strat][0]:.2f}% ± {results[strat][1]:.2f}\n")

    print("="*55)
    print("📊 消融实验最终结果 (Mean ± Std, %)")
    print("="*55)
    print(f"{'Strategy':<15} | {'Mean±Std':<15}")
    print("-"*55)
    for strat, (m, s) in results.items():
        tag = " 🏆 (Ours)" if strat == "Combined" else ""
        print(f"{strat:<15} | {m:>6.2f} ± {s:<5.2f}{tag}")
    print("="*55)
    print(f"⏱️ 总耗时: {time.time()-total_start:.1f}s")