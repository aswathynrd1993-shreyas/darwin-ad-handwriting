"""
Leakage-safe nested cross-validation analysis of the DARWIN handwriting dataset.

Usage:  python darwin_nested_cv.py data.csv out_dir

Protocol
- Outer loop: repeated stratified 5-fold CV (5 repeats, 25 outer folds), used only for evaluation.
- Inner loop: stratified 5-fold CV on the outer-training part only, used for feature selection
  size and hyperparameter selection (grid search).
- All preprocessing (signed-log transform, K-FFS Fisher-score selection, standardisation)
  is inside a single sklearn Pipeline and therefore fitted on training data only.
- Models: Logistic Regression (LR), Random Forest (RF), Gradient Boosting (GB, sklearn
  HistGradientBoostingClassifier) and a soft-voting ensemble (mean of the three tuned models'
  predicted probabilities, ENS).
"""
import sys, os, json, re, itertools, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_REPEATS, N_OUTER, N_INNER = 5, 5, 5
N_BOOT = 2000


# ---------------------------------------------------------------- K-FFS selector
def fisher_score(X, y):
    a, b = X[y == 0], X[y == 1]
    num = np.abs(a.mean(0) - b.mean(0))
    den = np.sqrt(a.var(0) + b.var(0)) + 1e-12
    return num / den


class KFFS(BaseEstimator, TransformerMixin):
    """k-fold feature selection: the Fisher-score filter is applied to the training part of
    each of k folds of the data passed to fit(); a feature is kept if it is in the top
    `n_top` features in at least a fraction `th` of the folds."""
    def __init__(self, n_top=50, k=5, th=0.8, random_state=SEED):
        self.n_top, self.k, self.th, self.random_state = n_top, k, th, random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        counts = np.zeros(X.shape[1])
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.random_state)
        for tr, _ in skf.split(X, y):
            s = fisher_score(X[tr], y[tr])
            counts[np.argsort(-s)[: self.n_top]] += 1
        self.freq_ = counts / self.k
        sel = np.where(self.freq_ >= self.th)[0]
        if len(sel) == 0:
            sel = np.argsort(-counts)[: self.n_top]
        self.support_ = np.sort(sel)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def slog(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_pipes():
    pre = [("slog", FunctionTransformer(slog)), ("kffs", KFFS())]
    lr = Pipeline(pre + [("sc", StandardScaler()),
                         ("clf", LogisticRegression(solver="lbfgs", max_iter=5000))])
    rf = Pipeline(pre + [("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))])
    gb = Pipeline(pre + [("clf", HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                                                early_stopping=False))])
    grids = {
        "LR": {"kffs__n_top": [20, 50, 100], "clf__C": [0.01, 0.1, 1.0, 10.0]},
        "RF": {"kffs__n_top": [20, 50, 100], "clf__max_features": ["sqrt", 0.2],
               "clf__min_samples_leaf": [1, 3]},
        "GB": {"kffs__n_top": [20, 50, 100], "clf__learning_rate": [0.03, 0.1],
               "clf__max_depth": [2, 3]},
    }
    return {"LR": lr, "RF": rf, "GB": gb}, grids


# ---------------------------------------------------------------- metrics
def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(accuracy=(tp + tn) / len(y), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=tp / (tp + fp) if tp + fp else np.nan, f1=f1_score(y, yhat),
                auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), TP=int(tp), TN=int(tn),
                FP=int(fp), FN=int(fn))


def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(penalty=None).fit(lg, y)
    slope = m.coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model)
    from scipy.optimize import minimize_scalar
    nll = lambda a: -np.sum(y * (a + lg[:, 0]) - np.log1p(np.exp(a + lg[:, 0])))
    a = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
    return dict(cal_intercept=a, cal_slope=slope)


def sens_at_spec(y, p, spec=0.9):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_ci(y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "brier"]
    out = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = metrics(y[i], p[i])
        for k in keys:
            out[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(d.mean()), float(t), float(p)


def mcnemar(y, pa, pb):
    ca, cb = (pa >= .5) == y, (pb >= .5) == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)


def nogueira_stability(Z):
    """Nogueira et al. (2018) stability of feature selection; Z: (M runs x d) binary."""
    Z = np.asarray(Z, float)
    M, d = Z.shape
    pf = Z.mean(0)
    kbar = Z.sum(1).mean()
    num = (M / (M - 1)) * np.mean(pf * (1 - pf))
    den = (kbar / d) * (1 - kbar / d)
    return float(1 - num / den) if den > 0 else np.nan


# ---------------------------------------------------------------- main
def main(path, out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d = pd.read_csv(path)
    feats = [c for c in d.columns if c not in ("ID", "class")]
    X = d[feats].to_numpy(float)
    y = (d["class"] == "P").astype(int).to_numpy()
    n = len(y)
    info = dict(n_subjects=int(n), n_AD=int(y.sum()), n_HC=int((1 - y).sum()), n_columns=int(d.shape[1]),
                n_features=len(feats), n_tasks=25, features_per_task=18, missing=int(d.isna().sum().sum()))
    print(info)

    pipes, grids = make_pipes()
    names = list(pipes) + ["ENS"]
    outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    fold_rows, sel_masks, best_params = [], {k: [] for k in pipes}, {k: [] for k in pipes}
    oof = {k: np.zeros((N_REPEATS, n)) for k in names}
    perm_imp = {k: [] for k in pipes}

    for f, (tr, te) in enumerate(outer.split(X, y)):
        rep = f // N_OUTER
        probs = {}
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED + f)
        for name, pipe in pipes.items():
            gs = GridSearchCV(pipe, grids[name], scoring="roc_auc", cv=inner, n_jobs=2, refit=True)
            gs.fit(X[tr], y[tr])
            p = gs.predict_proba(X[te])[:, 1]
            probs[name] = p
            best_params[name].append({k: (v if not isinstance(v, np.generic) else v.item())
                                      for k, v in gs.best_params_.items()})
            mask = np.zeros(len(feats), bool)
            mask[gs.best_estimator_.named_steps["kffs"].support_] = True
            sel_masks[name].append(mask)
            # permutation importance on the outer test fold; only selected features can matter,
            # unselected features have importance exactly 0 by construction
            imp = np.zeros(len(feats))
            base = roc_auc_score(y[te], p)
            rng = np.random.default_rng(SEED + f)
            for j in np.where(mask)[0]:
                drops = []
                for _ in range(5):
                    Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                    drops.append(base - roc_auc_score(y[te], gs.best_estimator_.predict_proba(Xp)[:, 1]))
                imp[j] = np.mean(drops)
            perm_imp[name].append(imp)
        probs["ENS"] = np.mean([probs[k] for k in pipes], axis=0)
        for name in names:
            oof[name][rep, te] = probs[name]
            m = metrics(y[te], probs[name])
            fold_rows.append(dict(model=name, repeat=rep + 1, fold=f % N_OUTER + 1, n_test=len(te), **m))
        print(f"outer fold {f + 1}/{N_REPEATS * N_OUTER} done  ({time.time() - t0:.0f}s)", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(f"{out}/fold_level_metrics.csv", index=False)

    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "brier"]
    summary = {}
    for name in names:
        fd = folds[folds.model == name]
        p_avg = oof[name].mean(0)
        pooled = metrics(y, p_avg)
        per_rep = [metrics(y, oof[name][r]) for r in range(N_REPEATS)]
        summary[name] = dict(
            fold_mean={k: float(fd[k].mean()) for k in keys},
            fold_sd={k: float(fd[k].std(ddof=1)) for k in keys},
            pooled_oof=pooled,
            pooled_ci95=boot_ci(y, p_avg),
            per_repeat_confusion=[{k: r[k] for k in ("TP", "TN", "FP", "FN")} for r in per_rep],
            per_repeat_accuracy=[float(r["accuracy"]) for r in per_rep],
            sens_at_spec90=sens_at_spec(y, p_avg, 0.9),
            spec_at_sens90=sens_at_spec(1 - y, 1 - p_avg, 0.9),
            **calib(y, p_avg),
        )
    pd.DataFrame({k: oof[k].mean(0) for k in names} | {"ID": d["ID"], "y": y}).to_csv(
        f"{out}/oof_probabilities_mean_over_repeats.csv", index=False)

    # pairwise significance
    n_test = n / N_OUTER
    n_train = n - n_test
    sig = []
    for a, b in itertools.combinations(names, 2):
        for metric in ("auc", "accuracy"):
            va = folds[folds.model == a].sort_values(["repeat", "fold"])[metric].to_numpy()
            vb = folds[folds.model == b].sort_values(["repeat", "fold"])[metric].to_numpy()
            diff, t, p = corrected_ttest(va, vb, n_train, n_test)
            sig.append(dict(A=a, B=b, metric=metric, mean_diff=diff, t=t, p_corrected_ttest=p))
        bb, cc, pm = mcnemar(y, oof[a].mean(0), oof[b].mean(0))
        sig.append(dict(A=a, B=b, metric="mcnemar(pooled)", mean_diff=np.nan, t=np.nan,
                        p_corrected_ttest=pm, A_only_correct=bb, B_only_correct=cc))
    pd.DataFrame(sig).to_csv(f"{out}/pairwise_significance.csv", index=False)

    # feature selection stability and frequency
    fs = {}
    freq_tab = pd.DataFrame(index=feats)
    for name in pipes:
        Z = np.array(sel_masks[name])
        fs[name] = dict(n_selected_mean=float(Z.sum(1).mean()), n_selected_sd=float(Z.sum(1).std(ddof=1)),
                        n_selected_min=int(Z.sum(1).min()), n_selected_max=int(Z.sum(1).max()),
                        n_selected_per_fold=Z.sum(1).tolist(), nogueira_stability=nogueira_stability(Z),
                        best_params_counts=pd.Series([json.dumps(b, sort_keys=True) for b in best_params[name]])
                        .value_counts().to_dict())
        freq_tab[f"{name}_selection_freq"] = Z.mean(0)
        I = np.array(perm_imp[name])
        freq_tab[f"{name}_perm_importance_mean"] = I.mean(0)
        freq_tab[f"{name}_perm_importance_sd"] = I.std(0, ddof=1)
        # rank stability of importance: mean pairwise Spearman between folds (features selected >= once)
        top = np.argsort(-I, axis=1)[:, :10]
        topcount = np.zeros(len(feats))
        for row in top:
            topcount[row] += 1
        freq_tab[f"{name}_top10_freq"] = topcount / len(I)
        rho = [stats.spearmanr(I[i], I[j]).statistic for i, j in itertools.combinations(range(len(I)), 2)]
        fs[name]["importance_rank_spearman_mean"] = float(np.nanmean(rho))
        fs[name]["importance_rank_spearman_sd"] = float(np.nanstd(rho))
    freq_tab.to_csv(f"{out}/feature_selection_and_importance.csv")
    per_fold = []
    for name in pipes:
        for f_i, (mask, bp) in enumerate(zip(sel_masks[name], best_params[name])):
            per_fold.append(dict(model=name, repeat=f_i // N_OUTER + 1, fold=f_i % N_OUTER + 1,
                                 best_params=json.dumps(bp, sort_keys=True), n_selected=int(mask.sum()),
                                 selected_features=";".join(np.array(feats)[mask])))
    pd.DataFrame(per_fold).to_csv(f"{out}/per_fold_selected_features_and_hyperparameters.csv", index=False)

    # ---------------- task-wise analysis (18 features per task, fixed hyperparameters, no FS)
    task_rows = []
    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rskf.split(X, y))
    tmods = {
        "LR": Pipeline([("slog", FunctionTransformer(slog)), ("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=0.1, max_iter=5000))]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2),
        "GB": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=2,
                                             random_state=SEED, early_stopping=False),
    }
    for t in range(1, 26):
        cols = [i for i, c in enumerate(feats) if re.search(rf"\D{t}$", c)]
        assert len(cols) == 18, (t, len(cols))
        Xt = X[:, cols]
        res = {k: [] for k in list(tmods) + ["ENS"]}
        for tr, te in splits:
            ps = {}
            for k, m in tmods.items():
                mm = clone(m).fit(Xt[tr], y[tr])
                ps[k] = mm.predict_proba(Xt[te])[:, 1]
            ps["ENS"] = np.mean(list(ps.values()), axis=0)
            for k, p in ps.items():
                res[k].append((accuracy_score(y[te], p >= .5), roc_auc_score(y[te], p)))
        for k, v in res.items():
            v = np.array(v)
            task_rows.append(dict(task=t, model=k, acc_mean=v[:, 0].mean(), acc_sd=v[:, 0].std(ddof=1),
                                  auc_mean=v[:, 1].mean(), auc_sd=v[:, 1].std(ddof=1)))
        print(f"task {t} done ({time.time() - t0:.0f}s)", flush=True)
    tasks = pd.DataFrame(task_rows)
    tasks.to_csv(f"{out}/taskwise_results.csv", index=False)

    # ---------------- figures
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for name in names:
        p = oof[name].mean(0)
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
        fr, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax[1].plot(mp, fr, "o-", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC, pooled out-of-fold"); ax[0].legend(fontsize=8)
    ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed AD fraction")
    ax[1].set_title("(b) Calibration"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig_roc_calibration.png", dpi=300); plt.close()

    piv = tasks.pivot(index="task", columns="model", values="auc_mean")[["LR", "RF", "GB", "ENS"]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = .2
    for i, c in enumerate(piv.columns):
        ax.bar(piv.index + (i - 1.5) * w, piv[c], w, label=c)
    ax.set_xticks(range(1, 26)); ax.set_xlabel("Task"); ax.set_ylabel("Mean AUC (25 folds)"); ax.set_ylim(.5, 1)
    ax.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.savefig(f"{out}/fig_taskwise_auc.png", dpi=300); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, name in zip(ax, pipes):
        s = freq_tab[f"{name}_perm_importance_mean"].sort_values(ascending=False).head(15)[::-1]
        e = freq_tab.loc[s.index, f"{name}_perm_importance_sd"]
        a.barh(s.index, s.values, xerr=e.values, color="tab:blue", alpha=.8)
        a.set_title(f"{name}: permutation importance (ΔAUC)", fontsize=10); a.tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(f"{out}/fig_permutation_importance.png", dpi=300); plt.close()

    json.dump(dict(dataset=info, protocol=dict(seed=SEED, repeats=N_REPEATS, outer_folds=N_OUTER,
                                                inner_folds=N_INNER, bootstrap=N_BOOT, grids=grids),
                   summary=summary, feature_selection=fs, runtime_s=time.time() - t0),
              open(f"{out}/results.json", "w"), indent=2, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
