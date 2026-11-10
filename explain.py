"""Explainable AI: permutation importance, SHAP, prediction explanations, what-if."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

from .utils import show, need

TREE = ("Random Forest", "XGBoost", "Decision Tree")


def _names(pipe):
    try: return list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception: return None


def shap_matrix(pipe, name, Xraw, Xbg_raw):
    import shap
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt, Xb = prep.transform(Xraw), prep.transform(Xbg_raw)
    ex = shap.TreeExplainer(model) if name in TREE else shap.LinearExplainer(model, Xb)
    sv = ex.shap_values(Xt)
    if isinstance(sv, list): sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, -1]
    base = np.ravel(ex.expected_value)[-1]
    return sv, float(base), Xt


def _clean(n): return n.split("__", 1)[-1]


def render(df):
    st.header("🧠 Explainable AI")
    r = st.session_state.get("ml")
    need(r, "Train models first in **🤖 Machine Learning** – explanations are computed on those fitted pipelines.")
    name = st.selectbox("Model to explain", list(r["pipes"]), index=list(r["pipes"]).index(r["best"]))
    pipe = r["pipes"][name]; Xte, yte = r["X_test"], r["y_test"]; task = r["task"]
    tabs = st.tabs(["Feature importance", "SHAP global", "Prediction explanation", "What-if"])

    with tabs[0]:
        with st.spinner("Computing permutation importance on held-out data..."):
            n = min(len(Xte), 1000); idx = np.random.default_rng(0).choice(len(Xte), n, replace=False)
            pi = permutation_importance(pipe, Xte.iloc[idx], yte.iloc[idx], n_repeats=5, random_state=42, n_jobs=1,
                                        scoring="r2" if task == "regression" else "f1_weighted")
        imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance")
        show(px.bar(imp.tail(20), x="importance", y="feature", orientation="h", error_x="std"), "Permutation importance (original features)", height=520)
        st.caption("Drop in test score when a feature is shuffled – model-agnostic and computed on unseen data.")
        m = pipe.named_steps["model"]; nm = _names(pipe)
        vals = getattr(m, "feature_importances_", None)
        if vals is None and hasattr(m, "coef_"): vals = np.abs(np.ravel(m.coef_)) if np.ndim(m.coef_) <= 1 else np.abs(m.coef_).mean(axis=0)
        if vals is not None and nm is not None and len(vals) == len(nm):
            fi = pd.DataFrame({"feature": [_clean(x) for x in nm], "value": vals}).sort_values("value").tail(20)
            show(px.bar(fi, x="value", y="feature", orientation="h"), "Model-native importance (encoded features)", height=520)

    try:
        import shap  # noqa
    except Exception:
        for t in tabs[1:3]: t.error("The `shap` package is not installed.")
        return
    nm = [_clean(x) for x in (_names(pipe) or [])]
    bg = Xte.sample(min(100, len(Xte)), random_state=0); smp = Xte.sample(min(300, len(Xte)), random_state=1)
    try:
        sv, base, Xt = shap_matrix(pipe, name, smp, bg)
    except Exception as e:
        tabs[1].error(f"SHAP failed for this model: {e}"); return

    with tabs[1]:
        mean_abs = np.abs(sv).mean(axis=0)
        gi = pd.DataFrame({"feature": nm, "mean|SHAP|": mean_abs}).sort_values("mean|SHAP|").tail(15)
        show(px.bar(gi, x="mean|SHAP|", y="feature", orientation="h"), "Global SHAP importance", height=480)
        top = gi.feature.tolist()[::-1][:8]; rows = []
        rng = np.random.default_rng(0)
        for f in top:
            j = nm.index(f); v = Xt[:, j]; sc = (v - v.min()) / (np.ptp(v) or 1)
            rows.append(pd.DataFrame({"feature": f, "shap": sv[:, j], "norm_value": sc, "jit": rng.uniform(-.3, .3, len(v))}))
        bs = pd.concat(rows); bs["y"] = bs.feature.map({f: i for i, f in enumerate(top[::-1])}) + bs.jit
        fig = go.Figure(go.Scatter(x=bs.shap, y=bs.y, mode="markers", marker=dict(color=bs.norm_value, colorscale="RdBu_r", size=5, showscale=True, colorbar=dict(title="feature value"))))
        fig.update_yaxes(tickvals=list(range(len(top))), ticktext=top[::-1])
        show(fig, "SHAP summary (beeswarm)", height=480)
        f = st.selectbox("Dependence plot for", [x for x in nm], index=nm.index(top[0]))
        j = nm.index(f); show(px.scatter(x=Xt[:, j], y=sv[:, j], labels={"x": f"{f} (encoded)", "y": "SHAP value"}, opacity=.6), f"SHAP dependence – {f}")

    with tabs[2]:
        i = st.number_input("Row of the test set", 0, len(Xte) - 1, 0)
        row = Xte.iloc[[int(i)]]; s1, b1, X1 = shap_matrix(pipe, name, row, bg)
        pred = pipe.predict(row)[0]
        if task == "classification":
            lab = r["label_encoder"].inverse_transform([int(pred)])[0]; pr = pipe.predict_proba(row)[0]
            st.write(f"Predicted: **{lab}** (confidence {pr.max():.1%}) · actual: **{r['label_encoder'].inverse_transform([int(yte.iloc[int(i)])])[0]}**")
        else:
            st.write(f"Predicted: **{pred:,.3f}** · actual: **{yte.iloc[int(i)]:,.3f}**")
        c = pd.DataFrame({"feature": nm, "contribution": s1[0]}); c["abs"] = c.contribution.abs()
        c = c.sort_values("abs").tail(12)
        show(px.bar(c, x="contribution", y="feature", orientation="h", color=c.contribution > 0, color_discrete_map={True: "#ef4444", False: "#3b82f6"}), "Why this prediction? (SHAP contributions, red = pushes up)", height=450)
        st.caption(f"Base value {b1:.3f} + sum of contributions {s1[0].sum():+.3f} (model output space: {'log-odds / margin' if task=='classification' else 'target units'}).")
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}))

    with tabs[3]:
        st.write("Change the inputs and see the prediction and its drivers update.")
        vals = {}; cols = st.columns(3)
        for k, c in enumerate(r["num"] + r["cat"]):
            with cols[k % 3]:
                if c in r["num"]:
                    s = r["X_train"][c]; vals[c] = st.number_input(c, value=float(s.median()), key=f"wi_{c}")
                else:
                    opts = sorted(r["X_train"][c].dropna().unique().tolist()); vals[c] = st.selectbox(c, opts, key=f"wi_{c}")
        row = pd.DataFrame([vals])[r["X_train"].columns]
        pred = pipe.predict(row)[0]
        if task == "classification":
            pr = pipe.predict_proba(row)[0]; cls = r["label_encoder"].classes_
            show(px.bar(x=cls, y=pr, labels={"x": "class", "y": "probability"}), "Predicted class probabilities", height=300)
        else:
            st.metric("Prediction", f"{pred:,.3f}")
        s2, _, _ = shap_matrix(pipe, name, row, bg)
        c = pd.DataFrame({"feature": nm, "contribution": s2[0]}); c = c.reindex(c.contribution.abs().sort_values().index).tail(10)
        show(px.bar(c, x="contribution", y="feature", orientation="h"), "Drivers of this scenario", height=380)
