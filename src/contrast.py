"""Week 11 诊断: CV轨迹 + 极刑轨迹 + 极刑集合Jaccard
区分 'impact≈位置(信号同构)' vs '都惩罚不足(都失效)'"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128; EPOCHS=10; LR=0.001
P_EXTREME=0.8; TAU_DIR=0.7; LOW_PCT=30
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
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
analyze_loader=DataLoader(train_ds,batch_size=512,shuffle=False)

def analyze(model,strategy,num_neurons=256):
    """返回 (probs, n_ext, cv, ext_set)"""
    model.eval()
    probs=np.full(num_neurons,0.5)
    W2=model.fc2.weight.data.cpu(); static=torch.sum(torch.abs(W2),dim=0).numpy()
    cv=float(np.std(static)/np.mean(static))
    W1=model.fc1.weight.data.cpu(); W1n=F.normalize(W1,p=2,dim=1)
    sim=torch.mm(W1n,W1n.T).numpy(); np.fill_diagonal(sim,-1.0)
    max_sim=np.max(sim,axis=1)
    dyn=np.zeros(num_neurons); N=0
    with torch.no_grad():
        for i,(d,_) in enumerate(analyze_loader):
            if i>=ANALYZE_BATCHES: break
            d=d.to(device).view(-1,28*28)
            dyn+=torch.sum(model.relu1(model.fc1(d)),dim=0).cpu().numpy(); N+=d.shape[0]
    pos_emb=dyn/N; impact=dyn*static
    dir_mask=max_sim>TAU_DIR
    ext=set()
    if strategy=="Dir+PosEmb":
        idx=np.where(dir_mask & (pos_emb<np.percentile(pos_emb,LOW_PCT)))[0]
    elif strategy=="Dir+Func":
        idx=np.where(dir_mask & (impact<np.percentile(impact,LOW_PCT)))[0]
    else:
        idx=np.where(dir_mask)[0]
    ext=set(idx.tolist()); probs[list(ext)]=P_EXTREME
    return torch.tensor(probs,dtype=torch.float32),len(ext),cv,ext

def run_diag(strategy,seed=42):
    set_seed(seed)
    model=HyperplaneMLP().to(device)
    crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LR)
    traj=[]
    for epoch in range(1,EPOCHS+1):
        if epoch>WARMUP and (epoch-WARMUP)%UPDATE_INTERVAL==1:
            newp,n_ext,cv,ext=analyze(model,strategy)
            traj.append((epoch,cv,n_ext,ext))
            old=model.hp1.dropout_probs.cpu()
            model.hp1.update_probs(SMOOTH_ALPHA*newp+(1-SMOOTH_ALPHA)*old)
        model.train()
        for d,t in train_loader:
            d,t=d.to(device),t.to(device)
            opt.zero_grad(); loss=crit(model(d),t); loss.backward(); opt.step()
    return traj

if __name__=="__main__":
    tA=run_diag("Dir+PosEmb"); tB=run_diag("Dir+Func"); tC=run_diag("Dir-Only")
    print(f"{'epoch':<6}|{'PosEmb cv/n':<16}|{'Func cv/n':<16}|{'DirOnly n':<10}| Jaccard(Pos,Func)")
    print("-"*70)
    for (eA,cvA,nA,sA),(eB,cvB,nB,sB),(eC,cvC,nC,sC) in zip(tA,tB,tC):
        jac=len(sA&sB)/max(1,len(sA|sB))
        print(f"{eA:<6}|{cvA:.3f}/{nA:<10}|{cvB:.3f}/{nB:<10}|{nC:<10}| {jac:.2f}")