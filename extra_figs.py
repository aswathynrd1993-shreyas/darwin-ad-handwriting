import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d=pd.read_csv(sys.argv[1]); out=sys.argv[2]
feats=[c for c in d.columns if c not in("ID","class")]
X=d[feats]; y=(d["class"]=="P").astype(int)
a,b=X[y==0],X[y==1]; fs=(a.mean()-b.mean()).abs()/np.sqrt(a.var()+b.var())
top=fs.sort_values(ascending=False).head(20).index
C=X[top].corr(method="spearman")
fig,ax=plt.subplots(figsize=(8,6.8)); im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(20)); ax.set_xticklabels(top,rotation=90,fontsize=7); ax.set_yticks(range(20)); ax.set_yticklabels(top,fontsize=7)
plt.colorbar(im,fraction=.046); ax.set_title("Spearman correlation of the 20 features with highest Fisher score (full data, descriptive only)",fontsize=8)
plt.tight_layout(); plt.savefig(f"{out}/fig_corr_heatmap.png",dpi=300); plt.close()
# pipeline diagram
fig,ax=plt.subplots(figsize=(11,2.0)); ax.axis("off")
boxes=["DARWIN\n174 subjects\n450 features","Outer loop\nrepeated stratified\n5-fold (5×)","Inner loop (training\npart only): signed-log →\nK-FFS (Fisher) →\nscaling → grid search","Classifiers\nLR · RF · GB\n+ soft-voting ENS","Evaluation on\nheld-out outer fold\n(metrics, CI,\ncalibration)","Explanation\npermutation importance,\nselection frequency,\nstability"]
w=1.55;g=.25
for i,t in enumerate(boxes):
    x=i*(w+g); ax.add_patch(FancyBboxPatch((x,.35),w,1.3,boxstyle="round,pad=0.03",fc="#e8eef7",ec="#2b5c9e"))
    ax.text(x+w/2,1.0,t,ha="center",va="center",fontsize=8.5)
    if i<len(boxes)-1: ax.annotate("",xy=(x+w+g,1.0),xytext=(x+w,1.0),arrowprops=dict(arrowstyle="->",color="#2b5c9e"))
ax.set_xlim(-.1,len(boxes)*(w+g)); ax.set_ylim(0,2)
plt.tight_layout(); plt.savefig(f"{out}/fig_pipeline.png",dpi=300); plt.close()
print(list(top[:5]))
