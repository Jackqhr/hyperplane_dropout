"""Week 7 Stress2: 标签噪声 + 弱正则 + 量化指标, 拉开三栏差距"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SAMPLES=300; NOISE=0.30; LABEL_FLIP=0.08   # 8% 标签噪声
HIDDEN1,HIDDEN2=256,128
WEAK_P=0.2            # 弱基础正则 (Fixed 与 Ours 共用, 保证公平)
EPOCHS=500; LR=0.01; SEED=42

def set_seed(s): torch.manual_seed(s); np.random.seed(s)

class HyperplaneDropout(nn.Module):
    def __init__(self,n,default_p=0.5):
        super().__init__(); self.register_buffer('dropout_probs',torch.full((n,),float(default_p)))
    def update_probs(self,p): self.dropout_probs=p.to(self.dropout_probs.device)
    def forward(self,x):
        if not self.training: return x
        mask=(torch.rand_like(x)>self.dropout_probs).float()
        return x*mask/(1.0-self.dropout_probs+1e-6)

class MLP_2D(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        self.fc1=nn.Linear(2,HIDDEN1); self.relu1=nn.ReLU()
        self.drop1 = HyperplaneDropout(HIDDEN1,WEAK_P) if mode=="hyper" else (nn.Dropout(WEAK_P) if mode=="fixed" else nn.Identity())
        self.fc2=nn.Linear(HIDDEN1,HIDDEN2); self.relu2=nn.ReLU()
        self.drop2 = nn.Dropout(WEAK_P) if mode!="none" else nn.Identity()
        self.fc3=nn.Linear(HIDDEN2,2)
    def forward(self,x):
        x=self.drop1(self.relu1(self.fc1(x)))
        x=self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)

def analyze_and_update(model,X):
    model.eval(); n=HIDDEN1
    W2=model.fc2.weight.data.cpu(); static=torch.sum(torch.abs(W2),dim=0).numpy()
    with torch.no_grad(): dyn=torch.sum(model.relu1(model.fc1(X)),dim=0).cpu().numpy()
    impact=dyn*static
    W1=model.fc1.weight.data.cpu(); W1n=F.normalize(W1,p=2,dim=1)
    sim=torch.mm(W1n,W1n.T).numpy(); np.fill_diagonal(sim,-1.0); max_sim=np.max(sim,axis=1)
    probs=np.full(n,WEAK_P); low=np.percentile(impact,30); high=np.percentile(impact,70)
    for i in range(n):
        if impact[i]<low and max_sim[i]>0.7: probs[i]=0.9
        elif impact[i]>high: probs[i]=0.05
    old=model.drop1.dropout_probs.cpu()
    model.drop1.update_probs(0.5*torch.tensor(probs,dtype=torch.float32)+0.5*old)

def train_model(model,loader,X,mode):
    crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LR)  # Adam 更快 memorize
    for ep in range(1,EPOCHS+1):
        if mode=="hyper" and ep>50 and ep%50==0: analyze_and_update(model,X)
        model.train()
        for d,t in loader:
            d,t=d.to(device),t.to(device)
            opt.zero_grad(); loss=crit(model(d),t); loss.backward(); opt.step()

def metrics_and_plot(ax,model,X,y,Xt,yt,title):
    model.eval()
    # 干净测试集准确率
    with torch.no_grad():
        acc=(model(Xt).argmax(1)==yt).float().mean().item()*100
    # 边界曲折度
    h=0.02
    xx,yy=np.meshgrid(np.arange(X[:,0].min()-0.5,X[:,0].max()+0.5,h),
                      np.arange(X[:,1].min()-0.5,X[:,1].max()+0.5,h))
    g=torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32).to(device)
    with torch.no_grad():
        P=F.softmax(model(g),dim=1)[:,1].cpu().numpy().reshape(xx.shape)
    Zp=(P>0.5).astype(int)
    ratio=(np.sum(Zp[:,1:]!=Zp[:,:-1])+np.sum(Zp[1:,:]!=Zp[:-1,:]))/Zp.size
    ax.contourf(xx,yy,P,cmap=plt.cm.RdBu,alpha=0.8)
    ax.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdBu,edgecolors='k',s=30)
    ax.set_title(f"{title}\nTestAcc={acc:.1f}%  BoundaryRatio={ratio:.4f}",fontsize=10,fontweight='bold')
    return acc,ratio

if __name__=="__main__":
    set_seed(SEED)
    X,y=make_moons(n_samples=N_SAMPLES,noise=NOISE,random_state=SEED)
    flip=np.random.choice(N_SAMPLES,size=int(LABEL_FLIP*N_SAMPLES),replace=False)
    y[flip]=1-y[flip]   # 注入标签噪声
    Xt_,yt_=make_moons(1000,noise=0.25,random_state=999)  # 干净测试集
    Xtr=torch.tensor(X,dtype=torch.float32).to(device); ytr=torch.tensor(y,dtype=torch.long).to(device)
    Xte=torch.tensor(Xt_,dtype=torch.float32).to(device); yte=torch.tensor(yt_,dtype=torch.long).to(device)
    loader=DataLoader(TensorDataset(Xtr,ytr),batch_size=32,shuffle=True)

    fig,axes=plt.subplots(1,3,figsize=(18,5.5)); report={}
    for ax,mode,title in zip(axes,["none","fixed","hyper"],
        ["No Dropout","Fixed 0.2 (Baseline)","Ours (Adaptive)"]):
        set_seed(SEED); m=MLP_2D(mode).to(device)
        print(f"⏳ 训练 [{mode}] ..."); train_model(m,loader,Xtr,mode)
        report[mode]=metrics_and_plot(ax,m,X,y,Xte,yte,title)
    plt.tight_layout(); p=CKPT_DIR/"stress2_boundary.png"
    plt.savefig(p,dpi=300); print(f"✅ 已保存: {p}"); plt.show()
    print("\n📊 量化对比 (TestAcc%, BoundaryRatio↓):")
    for k,(a,r) in report.items(): print(f"  {k:<8} acc={a:.2f}  ratio={r:.5f}")