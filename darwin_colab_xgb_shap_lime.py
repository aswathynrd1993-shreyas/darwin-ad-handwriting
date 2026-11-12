"""
DARWIN - XGBoost with Optuna (nested), SHAP and LIME stability analysis.
Run in Google Colab:
    !pip install xgboost optuna shap lime
    # upload data.csv to the Colab session, then:
    !python darwin_colab_xgb_shap_lime.py data.csv out_xgb
Then download the folder out_xgb (zip it with: !zip -r out_xgb.zip out_xgb) and send it back.

Same leakage-safe protocol as darwin_nested_cv.py:
  outer repeated stratified 5-fold (5 repeats) for evaluation only;
  inner stratified 5-fold on the outer-training part for Optuna (TPE) tuning of
  K-FFS size + XGBoost hyperparameters; everything fitted on training folds only.
"""
import sys, os, json, time, itertools
import numpy as np, pandas as pd
import optuna, shap, xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, brier_score_loss
from sklearn.base import BaseEstimator, TransformerMixin

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED, R, K, KI, N_TRIALS = 42, 5, 5, 5, 40


def fisher_score(X, y):
    a, b = X[y == 0], X[y == 1]
    return np.abs(a.mean(0) - b.mean(0)) / (np.sqrt(a.var(0) + b.var(0)) + 1e-12)


class KFFS(BaseEstimator, TransformerMixin):
    def __init__(self, n_top=50, k=5, th=0.8, random_state=SEED):
        self.n_top, self.k, self.th, self.random_state = n_top, k, th, random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        c = np.zeros(X.shape[1])
        for tr, _ in StratifiedKFold(self.k, shuffle=True, random_state=self.random_state).split(X, y):
            c[np.argsort(-fisher_score(X[tr], y[tr]))[: self.n_top]] += 1
        sel = np.where(c / self.k >= self.th)[0]
        self.support_ = np.sort(sel if len(sel) else np.argsort(-c)[: self.n_top])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def slog(X):
    return np.sign(X) * np.log1p(np.abs(X))


def build(params):
    return Pipeline([("slog", FunctionTransformer(slog)), ("kffs", KFFS(n_top=params["n_top"])),
                     ("clf", xgb.XGBClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                               max_depth=params["max_depth"], min_child_weight=params["min_child_weight"],
                                               subsample=params["subsample"], colsample_bytree=params["colsample_bytree"],
                                               eval_metric="logloss", random_state=SEED, n_jobs=2))])


def metrics(y, p):
    yh = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(accuracy=(tp + tn) / len(y), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=tp / (tp + fp) if tp + fp else np.nan, f1=f1_score(y, yh), auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def main(path, out):
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(path)
    feats = [c for c in d.columns if c not in ("ID", "class")]
    X = d[feats].to_numpy(float); y = (d["class"] == "P").astype(int).to_numpy(); n = len(y)
    rows, oof, shap_imp, best = [], np.zeros((R, n)), [], []
    lime_rank_sets = []
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED).split(X, y)):
        inner = StratifiedKFold(KI, shuffle=True, random_state=SEED + f)

        def objective(trial):
            p = dict(n_top=trial.suggest_categorical("n_top", [20, 50, 100]),
                     n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
                     learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                     max_depth=trial.suggest_int("max_depth", 2, 4),
                     min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                     subsample=trial.suggest_float("subsample", 0.6, 0.9),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.6))
            return cross_val_score(build(p), X[tr], y[tr], cv=inner, scoring="roc_auc").mean()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + f))
        study.optimize(objective, n_trials=N_TRIALS)
        model = build(study.best_params).fit(X[tr], y[tr])
        best.append(study.best_params)
        p = model.predict_proba(X[te])[:, 1]
        oof[f // K, te] = p
        rows.append(dict(repeat=f // K + 1, fold=f % K + 1, **metrics(y[te], p)))

        # SHAP: TreeExplainer on the fitted XGBoost, background = outer-training data (transformed)
        Xtr_t = model[:-1].transform(X[tr]); Xte_t = model[:-1].transform(X[te])
        sel = [feats[i] for i in model.named_steps["kffs"].support_]
        ex = shap.TreeExplainer(model.named_steps["clf"], data=Xtr_t, feature_perturbation="interventional")
        sv = ex.shap_values(Xte_t)
        imp = pd.Series(np.abs(sv).mean(0), index=sel).reindex(feats).fillna(0.0)
        shap_imp.append(imp)

        # LIME: first test subject of each fold, 5 seeds x 5000 perturbations, top-10 overlap
        if f % K == 0:
            sets = []
            for s in range(5):
                le = LimeTabularExplainer(Xtr_t, feature_names=sel, class_names=["H", "AD"],
                                          discretize_continuous=True, random_state=s)
                e = le.explain_instance(Xte_t[0], model.named_steps["clf"].predict_proba, num_features=10,
                                        num_samples=5000)
                sets.append(set(name for name, _ in e.as_list()))
            jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
            lime_rank_sets.append(dict(fold=f + 1, subject_ID=d["ID"].iloc[te[0]], mean_jaccard_top10=float(np.mean(jac))))
        print(f"fold {f + 1}/{R * K}", flush=True)

    pd.DataFrame(rows).to_csv(f"{out}/xgb_fold_level_metrics.csv", index=False)
    pooled = metrics(y, oof.mean(0))
    S = pd.DataFrame(shap_imp)
    ranks = S.rank(axis=1, ascending=False)
    rho = [ranks.iloc[i].corr(ranks.iloc[j], method="spearman") for i, j in itertools.combinations(range(len(S)), 2)]
    summ = pd.DataFrame({"mean_abs_shap": S.mean(), "sd": S.std(), "top10_freq": (ranks <= 10).mean()}).sort_values(
        "mean_abs_shap", ascending=False)
    summ.to_csv(f"{out}/xgb_shap_importance_across_folds.csv")
    pd.DataFrame(lime_rank_sets).to_csv(f"{out}/lime_stability.csv", index=False)
    pd.DataFrame({"ID": d["ID"], "y": y, "XGB": oof.mean(0)}).to_csv(f"{out}/xgb_oof.csv", index=False)
    fd = pd.DataFrame(rows)
    json.dump(dict(pooled=pooled, fold_mean=fd.mean(numeric_only=True).to_dict(), fold_sd=fd.std(numeric_only=True).to_dict(),
                   shap_rank_spearman_mean=float(np.nanmean(rho)), best_params=best),
              open(f"{out}/xgb_results.json", "w"), indent=2, default=float)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
