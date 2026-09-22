import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np

# 锚定路径并导入模型
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from baseline import BaselineMLP, device, DATA_DIR

# ================= 1. 准备数据与模型 =================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# 这里为了统计快一点，我们可以只抽 10000 张图来统计激活率
subset_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
subset_loader = DataLoader(subset_dataset, batch_size=256, shuffle=False)

model = BaselineMLP().to(device)
model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "baseline_mnist.pth", map_location=device))
model.eval()

# ================= 2. 统计激活次数 =================
# 初始化计数器：fc1 有 256 个，fc2 有 128 个
fc1_activation_counts = torch.zeros(256).to(device)
fc2_activation_counts = torch.zeros(128).to(device)
total_samples = 0

print("正在遍历数据集，统计神经元激活状态...")
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(subset_loader):
        if batch_idx >= 40:  # 统计前 40 个 batch (约 10240 张图) 就足够说明问题了
            break

        data = data.to(device).view(-1, 28 * 28)
        total_samples += data.size(0)

        # 手动执行 forward，以便截取中间层的激活值
        out1 = model.relu1(model.fc1(data))
        out2 = model.relu2(model.fc2(out1))

        # 统计大于 0 的次数 (ReLU 激活)
        # out1 shape: [batch_size, 256] -> sum(dim=0) 变成 [256]
        fc1_activation_counts += (out1 > 0).sum(dim=0)
        fc2_activation_counts += (out2 > 0).sum(dim=0)

# ================= 3. 计算激活频率 =================
fc1_freq = (fc1_activation_counts / total_samples).cpu().numpy()
fc2_freq = (fc2_activation_counts / total_samples).cpu().numpy()

# ================= 4. 寻找“无意义/摸鱼”神经元 =================
# 定义：如果激活率低于 5%，我们认为这个超平面几乎是无效的
threshold = 0.05

dead_fc1 = np.sum(fc1_freq < threshold)
dead_fc2 = np.sum(fc2_freq < threshold)

print("\n" + "=" * 40)
print(f"📊 统计完成！(共分析 {total_samples} 张图像)")
print("-" * 40)
print(f"【fc1 层 (256个神经元)】:")
print(f" - 平均激活率: {np.mean(fc1_freq):.2%}")
print(f" - 🚨 极低激活率 (<{threshold:.0%}) 的'摸鱼'神经元数量: {dead_fc1} 个")

print(f"\n【fc2 层 (128个神经元)】:")
print(f" - 平均激活率: {np.mean(fc2_freq):.2%}")
print(f" - 🚨 极低激活率 (<{threshold:.0%}) 的'摸鱼'神经元数量: {dead_fc2} 个")
print("=" * 40)

# 打印出 fc2 层最摸鱼的 5 个神经元的激活率
print("\n🏆 fc2 层最'摸鱼'的 5 个神经元 (激活率最低):")
sorted_indices = np.argsort(fc2_freq)
for i in range(5):
    idx = sorted_indices[i]
    print(f"  神经元 #{idx}: 激活率仅为 {fc2_freq[idx]:.2%}")