import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

# 锚定路径并导入模型
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from baseline import BaselineMLP, device

# 加载模型
model = BaselineMLP().to(device)
model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "baseline_mnist.pth", map_location=device))
model.eval()

# ================= 提取 fc2 的权重 =================
# fc2 的权重 Shape 是 [128, 256] (128个神经元，每个接收fc1的256个输入)
W_fc2 = model.fc2.weight.data.cpu()
print(f"提取到 fc2 层权重，Shape: {W_fc2.shape}")

# ================= 计算余弦相似度 =================
W_normalized = F.normalize(W_fc2, p=2, dim=1)
similarity_matrix = torch.mm(W_normalized, W_normalized.T).numpy()

# ================= 统计高度相似的对数 =================
mask = np.ones_like(similarity_matrix, dtype=bool)
np.fill_diagonal(mask, False)
off_diag_values = similarity_matrix[mask]

high_sim_count = (off_diag_values > 0.8).sum()
very_high_sim_count = (off_diag_values > 0.9).sum()

print("\n" + "="*40)
print("📊 fc2 层 (深层) 静态几何冗余度统计：")
print("-" * 40)
# fc2 有 128 个神经元，总组合数是 128 * 127 = 16256 对
print(f"在 16256 对神经元组合中：")
print(f" - 🚨 相似度 > 0.8 的对数: {high_sim_count} 对")
print(f" - 🔥 极度相似 (> 0.9) 的对数: {very_high_sim_count} 对")
print("="*40)

if high_sim_count > 10:
    print("💡 结论：fc2 层存在明显的超平面平行/冗余现象！这就是我们要用 Dropout 惩罚的重点对象！")
else:
    print("💡 结论：连深层的静态权重都不怎么冗余，这个 Baseline 学得非常精简。")