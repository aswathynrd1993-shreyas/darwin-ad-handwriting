import json, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
r=json.load(open(sys.argv[1]))["summary"]
fig,ax=plt.subplots(figsize=(13.3,5)); ax.axis("off"); ax.set_xlim(0,13.3); ax.set_ylim(0,5)
ax.text(6.65,4.6,"Handwriting dynamics for classifying Alzheimer's disease: leakage-safe nested evaluation",ha="center",fontsize=15,weight="bold")
steps=[("DARWIN","174 participants\n89 AD / 85 HC\n25 tasks × 18 features"),
       ("Nested CV","outer: 5 × stratified 5-fold\ninner: 5-fold tuning\n(no test-data leakage)"),
       ("Pipeline","signed-log transform\nK-FFS Fisher selection\nLR · RF · GB · soft-voting"),
       ("Evaluation","mean ± SD, 95% CI,\nconfusion matrices,\ncalibration, tests"),
       ("Explanation","permutation importance\nselection frequency\nstability across folds")]
w=2.3; g=.32; x0=.25
for i,(h,t) in enumerate(steps):
    x=x0+i*(w+g)
    ax.add_patch(FancyBboxPatch((x,2.05),w,2.05,boxstyle="round,pad=0.04",fc="#e8eef7",ec="#2b5c9e",lw=1.5))
    ax.text(x+w/2,3.75,h,ha="center",fontsize=12.5,weight="bold",color="#1f3f6e")
    ax.text(x+w/2,2.95,t,ha="center",va="center",fontsize=9.8)
    if i<4: ax.annotate("",xy=(x+w+g-.02,3.05),xytext=(x+w+.02,3.05),arrowprops=dict(arrowstyle="-|>",color="#2b5c9e",lw=1.6))
rf=r["RF"]
res=(f"Best model (RF): accuracy {100*rf['fold_mean']['accuracy']:.1f} ± {100*rf['fold_sd']['accuracy']:.1f}%,  "
     f"AUC {rf['fold_mean']['auc']:.3f} ± {rf['fold_sd']['auc']:.3f}   |   "
     f"LR {100*r['LR']['fold_mean']['accuracy']:.1f}%, GB {100*r['GB']['fold_mean']['accuracy']:.1f}%, ENS {100*r['ENS']['fold_mean']['accuracy']:.1f}%")
ax.add_patch(FancyBboxPatch((.25,.35),12.8,1.35,boxstyle="round,pad=0.04",fc="#fdf3e3",ec="#c07a12",lw=1.5))
ax.text(6.65,1.3,res,ha="center",fontsize=12)
ax.text(6.65,.72,"Most influential feature families: total task time and in-air (pen-up) time  •  single tasks: AUC 0.70–0.88",ha="center",fontsize=11.5)
plt.savefig(sys.argv[2],dpi=300,bbox_inches="tight",pad_inches=.1); print("ok")
