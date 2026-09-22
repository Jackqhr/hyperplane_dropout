import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# 锚定路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from baseline import BaselineMLP, device, DATA_DIR

# ================= 1. 准备数据与模型 =================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
subset_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
subset_loader = DataLoader(subset_dataset, batch_size=256, shuffle=False)

model = BaselineMLP().to(device)
model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "baseline_mnist.pth", map_location=device))
model.eval()

# ================= 2. 提取静态连接强度 =================
# fc2 的权重 shape 是 [128, 256] (out_features, in_features)
# 每一列代表 fc1 中某个神经元连接到 fc2 所有 128 个神经元的权重
W2 = model.fc2.weight.data.cpu()

# 计算 fc1 每个神经元对下一层的“静态绝对连接强度”
# 对列求绝对值并相加，得到一个长度为 256 的向量
static_impact_fc1 = torch.sum(torch.abs(W2), dim=0).numpy()

# ================= 3. 计算动态边界贡献度 =================
# 我们需要在真实数据上，计算 激活值 * 静态强度
total_dynamic_impact = np.zeros(256)
total_samples = 0

print("正在遍历数据集，计算超平面真实破坏力...")
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(subset_loader):
        if batch_idx >= 40:  # 抽样 10000 张图计算
            break
        data = data.to(device).view(-1, 28 * 28)
        total_samples += data.size(0)

        # 获取 fc1 的激活值
        a1 = model.relu1(model.fc1(data))  # Shape: [batch_size, 256]

        # 计算该 batch 内的总贡献： sum(激活值) * 静态强度
        batch_impact = torch.sum(a1, dim=0).cpu().numpy() * static_impact_fc1
        total_dynamic_impact += batch_impact

# 归一化，得到每个神经元的平均贡献度得分
avg_impact_score = total_dynamic_impact / total_samples

# ================= 4. 寻找“狙击手”与“废话” =================
# 回顾 Week 3 的激活率 (这里为了演示，我们重新快速算一下 fc1 激活率)
fc1_freq = np.zeros(256)
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(subset_loader):
        if batch_idx >= 40: break
        data = data.to(device).view(-1, 28 * 28)
        a1 = model.relu1(model.fc1(data))
        fc1_freq += (a1 > 0).sum(dim=0).cpu().numpy()
fc1_freq /= (40 * 256)  # 粗略计算激活率

print("\n" + "=" * 40)
print("🕵️‍♂️ fc1 层神经元生态分析报告")
print("=" * 40)

# 找出“狙击手”：激活率极低 (<10%)，但贡献度极高 (排名前 20%)
threshold_freq = 0.10
threshold_impact = np.percentile(avg_impact_score, 80)

snipers = []
nonsense = []

for i in range(256):
    is_low_freq = fc1_freq[i] < threshold_freq
    is_high_impact = avg_impact_score[i] > threshold_impact
    is_low_impact = avg_impact_score[i] < np.percentile(avg_impact_score, 20)

    if is_low_freq and is_high_impact:
        snipers.append(i)
    elif is_low_impact:
        nonsense.append(i)

print(f"🔫 发现 【狙击手神经元】 (低激活，高贡献): {len(snipers)} 个")
print(f"   (它们平时摸鱼，但在决策边界处一锤定音！绝对不能 Drop！)")
if len(snipers) > 0:
    print(f"   代表神经元索引: {snipers[:5]}")

print(f"\n🗑️ 发现 【废话神经元】 (贡献度处于底部 20%): {len(nonsense)} 个")
print(f"   (它们对分类边界几乎没有实质性推动，是 Dropout 的首选目标！)")
if len(nonsense) > 0:
    print(f"   代表神经元索引: {nonsense[:5]}")
print("=" * 40)