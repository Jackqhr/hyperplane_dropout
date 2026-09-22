"""
Week 9: MNIST 升级版消融 (纯几何 vs 功能)
5策略: Baseline / Dir-Only / Func-Only / Dir+Func(旧) / Dir+Pos(新Ours)
统一"只惩罚不保护"对称结构, 唯一变量=第二维度(无/impact/alpha)
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np, time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 超参 (与旧实验一致, 保证可比) ----
BATCH_SIZE=128; EPOCHS=10; LR=0.001; SEEDS=[42,43,44]
P_EXTREME=0.8
TAU_DIR=0.7          # 方向阈值 (黄金阈值)
TAU_POS=0.9          # 位置阈值 (激活一致率, 默认0.9, 可调)
IMPACT_LOW_PCT=30
WARMUP=2; UPDATE_INTERVAL=2; SMOOTH_ALPHA=0.5; ANALYZE_BATCHES=10

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class HyperplaneDropout(nn.Module):
    def __init__(self,n,default_p=0.5):
        super().__init__(); self.register_buffer('dropout_probs',torch.full((n,),float(default_p)))
    def update_probs(self,p): self.dropout_probs=p.to(self.dropout_probs.device)
    def forward(self,x):
        if not self.training: return x
        mask=(torch.rand_like(x)>self.dropout_probs).float()
        return x*mask/(1.0-self.dropout_probs+1e-6)

class HyperplaneMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,256); self.relu1=nn.ReLU(); self.hp1=HyperplaneDropout(256)
        self.fc2=nn.Linear(256,128); self.relu2=nn.ReLU(); self.hp2=HyperplaneDropout(128)
        self.fc3=nn.Linear(128,10)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.hp1(self.relu1(self.fc1(x)))
        x=self.hp2(self.relu2(self.fc2(x)))
        return self.fc3(x)

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_ds=datasets.MNIST(root=str(DATA_DIR),train=True,download=True,transform=transform)
test_ds=datasets.MNIST(root=str(DATA_DIR),train=False,download=True,transform=transform)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_ds,batch_size=1000,shuffle=False)
analyze_loader=DataLoader(train_ds,batch_size=512,shuffle=False)

def analyze(model,strategy,num_neurons=256):
    """返回 (probs_tensor, 极刑数量)"""
    model.eval()
    probs=np.full(num_neurons,0.5)
    if strategy=="Baseline":
        return torch.tensor(probs,dtype=torch.float32),0

    # --- 方向冗余 max_sim ---
    W1=model.fc1.weight.data.cpu(); W1n=F.normalize(W1,p=2,dim=1)
    sim=torch.mm(W1n,W1n.T).numpy(); np.fill_diagonal(sim,-1.0)
    max_sim=np.max(sim,axis=1)

    # --- 位置冗余 max_alpha (跨batch累积) ---
    acc_on=np.zeros((num_neurons,num_neurons)); acc_off=np.zeros_like(acc_on); N=0
    with torch.no_grad():
        for i,(d,_) in enumerate(analyze_loader):
            if i>=ANALYZE_BATCHES: break
            d=d.to(device).view(-1,28*28)
            A=(model.fc1(d)>0).float().cpu().numpy()
            acc_on+=A.T@A; acc_off+=(1-A).T@(1-A); N+=A.shape[0]
    alpha=(acc_on+acc_off)/N; np.fill_diagonal(alpha,-1.0)
    max_alpha=np.max(alpha,axis=1)

    # --- 功能贡献 impact ---
    W2=model.fc2.weight.data.cpu(); static=torch.sum(torch.abs(W2),dim=0).numpy()
    dyn=np.zeros(num_neurons)
    with torch.no_grad():
        for i,(d,_) in enumerate(analyze_loader):
            if i>=ANALYZE_BATCHES: break
            d=d.to(device).view(-1,28*28)
            dyn+=torch.sum(model.relu1(model.fc1(d)),dim=0).cpu().numpy()
    impact=dyn*static
    low_thr=np.percentile(impact,IMPACT_LOW_PCT)

    # --- 按策略分配 (统一只惩罚) ---
    n_ext=0
    if strategy=="Dir-Only":
        m=max_sim>TAU_DIR; probs[m]=P_EXTREME; n_ext=m.sum()
    elif strategy=="Func-Only":
        m=impact<low_thr; probs[m]=P_EXTREME; n_ext=m.sum()
    elif strategy=="Dir+Func":
        for i in range(num_neurons):
            if max_sim[i]>TAU_DIR and impact[i]<low_thr: probs[i]=P_EXTREME; n_ext+=1
    elif strategy=="Dir+Pos":
        for i in range(num_neurons):
            if max_sim[i]>TAU_DIR and max_alpha[i]>TAU_POS: probs[i]=P_EXTREME; n_ext+=1
    return torch.tensor(probs,dtype=torch.float32),int(n_ext)

def run_one(strategy,seed):
    set_seed(seed)
    model=HyperplaneMLP().to(device)
    crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,EPOCHS+1):
        if epoch>WARMUP and (epoch-WARMUP)%UPDATE_INTERVAL==1:
            newp,n_ext=analyze(model,strategy)
            if epoch==WARMUP+1 and seed==SEEDS[0]:
                print(f"    [{strategy}] 首次更新极刑数={n_ext}")
            old=model.hp1.dropout_probs.cpu()
            model.hp1.update_probs(SMOOTH_ALPHA*newp+(1-SMOOTH_ALPHA)*old)
        model.train()
        for d,t in train_loader:
            d,t=d.to(device),t.to(device)
            opt.zero_grad(); loss=crit(model(d),t); loss.backward(); opt.step()
    model.eval(); c=tot=0
    with torch.no_grad():
        for d,t in test_loader:
            d,t=d.to(device),t.to(device)
            _,p=torch.max(model(d),1); tot+=t.size(0); c+=(p==t).sum().item()
    return 100*c/tot

if __name__=="__main__":
    strategies=["Baseline","Dir-Only","Func-Only","Dir+Func","Dir+Pos"]
    results={}; t0=time.time()
    for strat in strategies:
        accs=[run_one(strat,s) for s in SEEDS]
        results[strat]=(float(np.mean(accs)),float(np.std(accs)))
        print(f"✅ [{strat}] Mean={results[strat][0]:.2f}% ± {results[strat][1]:.2f}\n")
    print("="*55)
    print("📊 MNIST 升级版消融 (Mean±Std, %)")
    print("="*55)
    print(f"{'Strategy':<12} | {'Mean±Std':<15}")
    print("-"*55)
    for k,(m,s) in results.items():
        tag=" 🏆 (Ours, pure geometric)" if k=="Dir+Pos" else ""
        print(f"{k:<12} | {m:>6.2f} ± {s:<5.2f}{tag}")
    print("="*55)
    print(f"⏱️ 总耗时: {time.time()-t0:.1f}s")