import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# 锚定项目路径，以便导入同目录下的 baseline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from baseline import BaselineMLP, device

# ================= 1. 加载训练好的 Baseline 模型 =================
CKPT_DIR = PROJECT_ROOT / "checkpoints"
model_path = CKPT_DIR / "baseline_mnist.pth"

print(f"正在加载权重: {model_path}")
model = BaselineMLP().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 切换到评估模式

# ================= 2. 提取超平面法向量 =================
# 我们提取第一层 (fc1) 的权重。
# fc1 有 256 个神经元，输入是 784 维。
# 所以 weight 的 shape 是 [256, 784]。每一行代表一个神经元的超平面法向量。
W_fc1 = model.fc1.weight.data.cpu()
print(f"提取到 fc1 层权重，Shape: {W_fc1.shape}")

# ================= 3. 计算余弦相似度矩阵 =================
# 余弦相似度公式：cos(θ) = (A·B) / (|A| * |B|)
# 在 PyTorch 中，先对行向量进行 L2 归一化，然后做矩阵乘法即可得到相似度矩阵
W_normalized = F.normalize(W_fc1, p=2, dim=1) # Shape: [256, 784]

# 矩阵乘法: [256, 784] @ [784, 256] -> [256, 256]
similarity_matrix = torch.mm(W_normalized, W_normalized.T)
similarity_matrix = similarity_matrix.numpy()

print(f"相似度矩阵计算完成，Shape: {similarity_matrix.shape}")

# ================= 4. 可视化：寻找“冗余”的证据 =================
print("正在绘制热力图，请稍候...")
plt.figure(figsize=(10, 8))

# 使用 seaborn 画热力图
# cmap="coolwarm" 红色代表高度相似（冗余），蓝色代表不相似或负相关
sns.heatmap(similarity_matrix, cmap="coolwarm", center=0,
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine Similarity'})

plt.title("FC1 Layer Neuron Similarity Matrix\n(Red = Redundant/Parallel Hyperplanes)", fontsize=14)
plt.tight_layout()

# 保存图片
save_path = CKPT_DIR / "fc1_similarity_heatmap.png"
plt.savefig(save_path, dpi=300)
print(f"✅ 热力图已保存至: {save_path}")

# 顺便打印一下统计数据，看看“高度相似”的神经元有多少
# 排除对角线（自己和自己的相似度永远是1）
mask = np.ones_like(similarity_matrix, dtype=bool)
np.fill_diagonal(mask, False)
off_diag_values = similarity_matrix[mask]

high_sim_count = (off_diag_values > 0.7).sum() # 相似度大于 0.7 算高度相似
print(f"📊 统计: 在 {len(off_diag_values)} 对神经元组合中，有 {high_sim_count} 对的余弦相似度 > 0.7！")

# 显示图片 (如果你用的是 PyCharm 科学模式，或者在 Notebook 里)
plt.show()