import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import time

# ================= 0. 路径锚定（修复漂移坑） =================
# 当前文件是 src/baseline.py，往上两级就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # 数据集永远放这里
CKPT_DIR = PROJECT_ROOT / "checkpoints"   # 模型权重永远放这里
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 1. 基础设置 =================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ================= 2. 数据准备 (MNIST) =================
BATCH_SIZE = 128  # 针对 MX250 显存的安全值

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print(f"正在从 {DATA_DIR} 加载 MNIST 数据集...")
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ================= 3. 定义 Baseline 模型 =================
class BaselineMLP(nn.Module):
    def __init__(self):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

model = BaselineMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= 4. 训练与评估 =================
def train(epochs=10):
    print("\n--- 开始训练 Baseline 模型 ---")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch:02d}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}")

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# ================= 5. 主流程 =================
if __name__ == "__main__":
    start_time = time.time()
    train(epochs=10)
    acc = evaluate()
    elapsed = time.time() - start_time

    # 【新增】保存权重，供 Week 2 超平面分析直接加载
    save_path = CKPT_DIR / "baseline_mnist.pth"
    torch.save(model.state_dict(), save_path)

    print("\n" + "=" * 40)
    print("✅ Baseline 训练完成！")
    print(f"⏱️ 总耗时: {elapsed:.2f} 秒")
    print(f"🎯 测试集准确率: {acc:.2f}%")
    print(f"💾 权重已保存至: {save_path}")
    print("=" * 40)