"""Machine learning: leakage-safe pipelines, model comparison, CV, tuning, clustering."""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .utils import detect_types, show, need, kpi_row

PARAM_GRIDS = {
    "Random Forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 6, 12, 20], "model__min_samples_leaf": [1, 2, 5]},
    "XGBoost": {"model__n_estimators": [100, 200, 400], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.1, 0.2], "model__subsample": [0.7, 0.9, 1.0]},
    "Decision Tree": {"model__max_depth": [3, 5, 8, 12, None], "model__min_samples_leaf": [1, 5, 10, 20]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
}


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    return "classification" if y.nunique() <= 10 and (y.dropna() % 1 == 0).all() else "regression"


def get_models(task, balanced=False, fast=True):
    cw = "balanced" if balanced else None
    n = 150 if fast else 300
    if task == "regression":
        return {"Linear Regression": LinearRegression(), "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=n, learning_rate=.1, max_depth=5, n_jobs=-1, random_state=42)}
    return {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight=cw, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=n, class_weight=cw, n_jobs=-1, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=n, learning_rate=.1, max_depth=5, n_jobs=-1, random_state=42)}


def prepare_features(df, target, exclude=(), max_card=50):
    """Select usable features automatically; returns X, num_cols, cat_cols, dropped(reasons)."""
    types = detect_types(df)
    dropped = {}
    num, cat = [], []
    for c in df.columns:
        if c == target or c in exclude: continue
        if c in types["datetime"]: dropped[c] = "date column (use time-based split / feature engineering)"
        elif c in types["id"]: dropped[c] = "identifier-like column"
        elif df[c].nunique(dropna=True) <= 1: dropped[c] = "constant column"
        elif c in types["numeric"]: num.append(c)
        elif df[c].nunique() > max_card: dropped[c] = f"high-cardinality (> {max_card} levels)"
        else: cat.append(c)
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return X, num, cat, dropped


def leakage_suspects(df, target, num, thr=0.95):
    out = []
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    for c in num:
        d = pd.concat([df[c], y], axis=1).dropna()
        if len(d) > 10 and d.iloc[:, 0].nunique() > 1:
            r = abs(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
            if r >= thr: out.append((c, r))
    return out


def make_preprocessor(num, cat):
    t = []
    if num: t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat: t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))]), cat))
    return ColumnTransformer(t, remainder="drop")


def metrics(task, y_true, pred, proba=None):
    if task == "regression":
        return {"R2": r2_score(y_true, pred), "RMSE": np.sqrt(mean_squared_error(y_true, pred)), "MAE": mean_absolute_error(y_true, pred)}
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    m = {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, average=avg, zero_division=0),
         "Recall": recall_score(y_true, pred, average=avg, zero_division=0), "F1": f1_score(y_true, pred, average=avg, zero_division=0)}
    try:
        m["ROC_AUC"] = roc_auc_score(y_true, proba[:, 1]) if avg == "binary" else roc_auc_score(y_true, proba, multi_class="ovr")
    except Exception:
        m["ROC_AUC"] = np.nan
    return m


def train_and_compare(df, target, features_exclude=(), model_names=None, test_size=.2, cv=5, split="Random", date_col=None,
                      balanced=False, tune=False, n_iter=10, progress=None):
    """Leakage-safe training. Split BEFORE any fitting; all preprocessing lives inside the Pipeline."""
    d = df.dropna(subset=[target]).copy()
    task = infer_task(d[target])
    X, num, cat, dropped = prepare_features(d, target, features_exclude)
    need(num or cat, "No usable feature columns.")
    y_raw = d[target]; le = None
    y = y_raw
    if task == "classification":
        le = LabelEncoder(); y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
    if split == "Time-based" and date_col:
        order = d[date_col].argsort().values if hasattr(d[date_col], "argsort") else np.arange(len(d))
        cut = int(len(d) * (1 - test_size)); tr_i, te_i = order[:cut], order[cut:]
        Xtr, Xte, ytr, yte = X.iloc[tr_i], X.iloc[te_i], y.iloc[tr_i], y.iloc[te_i]
    else:
        strat = y if task == "classification" and y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    splitter = StratifiedKFold(cv, shuffle=True, random_state=42) if task == "classification" else KFold(cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_weighted"] if task == "classification" else ["r2", "neg_root_mean_squared_error"]
    models = get_models(task, balanced)
    models = {k: v for k, v in models.items() if not model_names or k in model_names}
    rows, fitted = [], {}
    for i, (name, est) in enumerate(models.items()):
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(num, cat)), ("model", est)])
        best = {}
        if tune and name in PARAM_GRIDS and not (task == "regression" and name == "Linear Regression"):
            s = RandomizedSearchCV(pipe, PARAM_GRIDS[name], n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1,
                                   scoring="f1_weighted" if task == "classification" else "r2")
            s.fit(Xtr, ytr); pipe = s.best_estimator_; best = {k.replace("model__", ""): v for k, v in s.best_params_.items()}
        cvr = cross_validate(pipe, Xtr, ytr, cv=splitter, scoring=scoring, n_jobs=1)   # CV on TRAIN only
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte); proba = pipe.predict_proba(Xte) if task == "classification" else None
        row = {"Model": name, **{f"{k}": v for k, v in metrics(task, yte, pred, proba).items()}}
        key = scoring[0]; row["CV_mean"] = cvr[f"test_{key}"].mean(); row["CV_std"] = cvr[f"test_{key}"].std()
        row["Train_time_s"] = time.time() - t0; row["Best_params"] = str(best) if best else "-"
        rows.append(row); fitted[name] = pipe
        if progress: progress.progress((i + 1) / len(models), f"Trained {name}")
    res = pd.DataFrame(rows)
    sort_col = "F1" if task == "classification" else "R2"
    res = res.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return dict(task=task, target=target, results=res, pipes=fitted, X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
                num=num, cat=cat, dropped=dropped, label_encoder=le, best=res.Model.iloc[0], sort_col=sort_col, split=split)


def model_diagnostics(r, name):
    pipe = r["pipes"][name]; Xte, yte = r["X_test"], r["y_test"]; pred = pipe.predict(Xte)
    if r["task"] == "regression":
        a, b = st.columns(2)
        with a:
            f = px.scatter(x=yte, y=pred, labels={"x": "Actual", "y": "Predicted"}, opacity=.5)
            f.add_shape(type="line", x0=float(yte.min()), y0=float(yte.min()), x1=float(yte.max()), y1=float(yte.max()), line=dict(dash="dash"))
            show(f, f"Actual vs predicted – {name}", height=380)
        with b: show(px.histogram(x=yte - pred, nbins=40, labels={"x": "Residual"}), "Residual distribution", height=380)
    else:
        classes = list(r["label_encoder"].classes_)
        cm = confusion_matrix(yte, pred)
        a, b = st.columns(2)
        with a: show(px.imshow(cm, x=classes, y=classes, text_auto=True, labels=dict(x="Predicted", y="Actual")), f"Confusion matrix – {name}", height=380)
        with b:
            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(yte, pipe.predict_proba(Xte)[:, 1])
                f = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC")); f.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="chance")
                show(f, f"ROC curve – {name}", height=380)


def render(df):
    st.header("🤖 Machine Learning")
    types = detect_types(df)
    tabs = st.tabs(["Supervised models", "Clustering (K-Means)"])
    with tabs[0]:
        cols = [c for c in df.columns if c not in types["id"]]
        target = st.selectbox("Target variable", cols, index=len(cols) - 1)
        task = infer_task(df[target].dropna()); st.info(f"Detected task: **{task}**")
        with st.expander("⚙️ Training settings", expanded=True):
            c1, c2, c3 = st.columns(3)
            test_size = c1.slider("Test size", .1, .4, .2, .05); cv = c2.slider("CV folds", 3, 10, 5)
            names = list(get_models(task)); chosen = c3.multiselect("Models", names, default=names)
            c4, c5, c6 = st.columns(3)
            split = c4.radio("Split", ["Random", "Time-based"], horizontal=True)
            date_col = c5.selectbox("Date column", types["datetime"]) if split == "Time-based" and types["datetime"] else None
            tune = c6.checkbox("Hyperparameter tuning (RandomizedSearchCV)")
            balanced = st.checkbox("Balance classes (class_weight)", value=False) if task == "classification" else False
            n_iter = st.slider("Tuning iterations", 5, 30, 10) if tune else 10
            avail = [c for c in df.columns if c != target]
            excl = st.multiselect("Exclude features (e.g. leakage columns)", avail)
        X, num, cat, dropped = prepare_features(df.dropna(subset=[target]), target, excl)
        sus = leakage_suspects(df, target, num)
        if sus:
            st.warning("⚠️ Possible **target leakage** – features almost perfectly correlated with the target: "
                       + ", ".join(f"`{c}` (|r|={r:.2f})" for c, r in sus) + ". Consider excluding them above.")
            if st.checkbox("Auto-exclude suspected leakage columns"): excl = list(excl) + [c for c, _ in sus]
        if dropped: st.caption("Auto-dropped: " + "; ".join(f"{k} ({v})" for k, v in dropped.items()))
        if split == "Time-based" and not date_col: st.warning("No date column available – using random split.")
        if st.button("🚀 Train & compare models", type="primary"):
            need(chosen, "Select at least one model.")
            bar = st.progress(0.0)
            try:
                with st.spinner("Training..."):
                    st.session_state["ml"] = train_and_compare(df, target, excl, chosen, test_size, cv, split, date_col, balanced, tune, n_iter, bar)
            except Exception as e:
                st.error(f"Training failed: {e}")
            bar.empty()
        r = st.session_state.get("ml")
        if r:
            st.success(f"Best model: **{r['best']}** (ranked by {r['sort_col']} on a held-out test set; preprocessing fitted on training data only)")
            show_df = r["results"].copy(); st.dataframe(show_df.round(4))
            mcols = [c for c in show_df.columns if c in ("R2", "RMSE", "MAE", "Accuracy", "F1", "ROC_AUC", "CV_mean")]
            show(px.bar(show_df.melt(id_vars="Model", value_vars=mcols), x="Model", y="value", color="variable", barmode="group"), "Model comparison")
            pick = st.selectbox("Inspect model", list(r["pipes"]), index=list(r["pipes"]).index(r["best"]))
            model_diagnostics(r, pick)
    with tabs[1]:
        num = types["numeric"]; need(len(num) >= 2, "Need ≥ 2 numeric columns.")
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        feats = st.multiselect("Features", num, default=num[:min(4, len(num))], key="km_f"); k = st.slider("Clusters (k)", 2, 10, 3, key="km_k")
        need(len(feats) >= 2, "Pick at least two features.")
        Z = StandardScaler().fit_transform(df[feats].fillna(df[feats].median()))
        lab = KMeans(k, n_init=10, random_state=42).fit_predict(Z); pc = PCA(2).fit_transform(Z)
        show(px.scatter(x=pc[:, 0], y=pc[:, 1], color=lab.astype(str), labels={"x": "PC1", "y": "PC2", "color": "cluster"}), "K-Means clusters (PCA)")
        out = df.copy(); out["cluster"] = lab; st.dataframe(out.groupby("cluster")[feats].mean().round(3))
        st.download_button("Download clustered data", out.to_csv(index=False).encode(), "clusters.csv")
