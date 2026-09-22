"""Week 7 Islands v2: 中间正则+高压力, 让孤岛指标有区分度 (均值仍严格校准)"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from scipy.ndimage import label
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN1,HIDDEN2=256,128
WEAK_P=0.1          # 中间正则: 让fixed有可见孤岛, 又给hyper打击空间
P_HIGH=0.7
N_SAMPLES=400; NOISE=0.35; FLIP_RATIO=0.12   # 高压力
EPOCHS=500; LR=0.01; SEEDS=[0,1,2]
ISLAND_THRS=[0.02,0.05,0.10]

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
        super().__init__()
        self.fc1=nn.Linear(2,HIDDEN1); self.relu1=nn.ReLU()
        self.drop1=HyperplaneDropout(HIDDEN1,WEAK_P) if mode=="hyper" else nn.Dropout(WEAK_P)
        self.fc2=nn.Linear(HIDDEN1,HIDDEN2); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(WEAK_P)
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

    low=np.percentile(impact,30); high=np.percentile(impact,70)
    protect=(impact>high); b=int(protect.sum())
    if b==0:
        idx=np.argsort(impact)[-10:]; protect=np.zeros(n,bool); protect[idx]=True; b=10
    # 候选惩罚按impact升序(最该死优先), 受a_max限制保证p_low>=0且均值=WEAK_P
    cand=np.where((impact<low)&(max_sim>0.7))[0]
    cand=cand[np.argsort(impact[cand])]
    a_max=int(WEAK_P*b/(P_HIGH-WEAK_P)); a=min(len(cand),a_max)
    probs=np.full(n,WEAK_P); probs[cand[:a]]=P_HIGH
    if a>0: probs[protect]=max(0.0, WEAK_P - a*(P_HIGH-WEAK_P)/b)
    print(f"    [update] a={a}/{len(cand)}, b={b}, mean_p={probs.mean():.3f}")
    old=model.drop1.dropout_probs.cpu()
    model.drop1.update_probs(0.5*torch.tensor(probs,dtype=torch.float32)+0.5*old)

def get_grid_pred(model,X,h=0.02):
    model.eval()
    xx,yy=np.meshgrid(np.arange(X[:,0].min()-0.5,X[:,0].max()+0.5,h),
                      np.arange(X[:,1].min()-0.5,X[:,1].max()+0.5,h))
    g=torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32).to(device)
    with torch.no_grad(): Zp=model(g).argmax(1).cpu().numpy().reshape(xx.shape)
    return Zp

def count_islands(Zp,thr):
    total=0
    for cls in [0,1]:
        lab,n=label(Zp==cls)
        if n<=1: continue
        sizes=np.bincount(lab.ravel())[1:]
        total+=int(np.sum(sizes<sizes.max()*thr))
    return total

def run(mode,seed):
    set_seed(seed)
    X,y=make_moons(n_samples=N_SAMPLES,noise=NOISE,random_state=seed)
    flip=np.random.choice(N_SAMPLES,size=int(FLIP_RATIO*N_SAMPLES),replace=False); y[flip]=1-y[flip]
    Xt,yt=make_moons(n_samples=1000,noise=0.25,random_state=999)
    Xtr=torch.tensor(X,dtype=torch.float32).to(device); ytr=torch.tensor(y,dtype=torch.long).to(device)
    Xte=torch.tensor(Xt,dtype=torch.float32).to(device); yte=torch.tensor(yt,dtype=torch.long).to(device)
    loader=DataLoader(TensorDataset(Xtr,ytr),batch_size=32,shuffle=True)
    m=MLP_2D(mode).to(device)
    opt=optim.Adam(m.parameters(),lr=LR); crit=nn.CrossEntropyLoss()
    for ep in range(1,EPOCHS+1):
        if mode=="hyper" and ep>50 and ep%50==0: analyze_and_update(m,Xtr)
        m.train()
        for d,t in loader:
            opt.zero_grad(); loss=crit(m(d),t); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): acc=(m(Xte).argmax(1)==yte).float().mean().item()*100
    Zp=get_grid_pred(m,X)
    return {t:count_islands(Zp,t) for t in ISLAND_THRS}, acc

if __name__=="__main__":
    print("📊 v2 高压力孤岛对比 (Mean±Std):")
    res={}
    for mode in ["fixed","hyper"]:
        isl={t:[] for t in ISLAND_THRS}; accs=[]
        for s in SEEDS:
            print(f"  [{mode}] seed={s}"); i,a=run(mode,s)
            for t in ISLAND_THRS: isl[t].append(i[t])
            accs.append(a)
        res[mode]=(isl,accs)
    print("\n"+"="*70)
    print(f"{'Mode':<7}|{'Acc%':<13}|"+"|".join(f"islands({t})" for t in ISLAND_THRS))
    print("-"*70)
    for mode,(isl,accs) in res.items():
        cells=[f"{np.mean(isl[t]):.2f}±{np.std(isl[t]):.2f}" for t in ISLAND_THRS]
        print(f"{mode:<7}|{np.mean(accs):.2f}±{np.std(accs):.2f} |"+"|".join(f"{c:^13}" for c in cells))
    print("="*70)