"""Automated insights: trends, anomalies, opportunities, recommendations."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from .utils import detect_types, outlier_mask, guess_col, show, human, cat_like

ICON = {"high": "🔴", "medium": "🟠", "low": "🟢", "info": "🔵"}


def _i(kind, sev, title, detail, action=""):
    return dict(type=kind, severity=sev, title=title, detail=detail, action=action)


def generate(df, ml_result=None):
    types = detect_types(df); num, dts, cats = types["numeric"], types["datetime"], cat_like(df, types)
    out = []
    # --- data quality
    miss = (100 * df.isna().mean()).sort_values(ascending=False); miss = miss[miss > 0]
    for c, p in miss.head(3).items():
        out.append(_i("Data quality", "high" if p > 20 else "medium" if p > 5 else "low", f"'{c}' has {p:.1f}% missing values",
                      "Missing data can bias statistics and models.", "Impute (median/mode) or drop the column in Data Cleaning."))
    dup = int(df.duplicated().sum())
    if dup: out.append(_i("Data quality", "medium", f"{dup:,} duplicate rows ({100 * dup / len(df):.1f}%)", "Duplicates inflate totals and counts.", "Remove duplicates."))
    # --- distributions & outliers
    for c in num:
        s = df[c].dropna()
        if len(s) > 20 and s.nunique() > 5:
            om = outlier_mask(s).mean()
            if om > .05: out.append(_i("Anomaly", "medium", f"'{c}' has {100 * om:.1f}% outliers (IQR rule)", "Heavy tails may distort means and linear models.", "Investigate, cap or use robust methods."))
            if abs(s.skew()) > 1.5: out.append(_i("Distribution", "low", f"'{c}' is strongly {'right' if s.skew() > 0 else 'left'}-skewed (skew {s.skew():.2f})", "Mean and median differ materially.", "Use median or a log transform."))
    # --- correlations
    if len(num) >= 2:
        c = df[num].corr().abs().where(lambda x: np.triu(np.ones(x.shape), 1).astype(bool)).stack().sort_values(ascending=False)
        for (a, b), r in c.head(3).items():
            if r > .8: out.append(_i("Relationship", "info", f"'{a}' and '{b}' are strongly correlated (|r| = {r:.2f})", "May indicate redundancy, derived fields or a driver relationship.", "Check for multicollinearity before modelling; avoid using derived fields as predictors."))
    # --- metric focus
    metric = guess_col(num, "revenue") or (num[0] if num else None)
    if metric and dts:
        s = df.set_index(dts[0])[metric].resample("MS").sum()
        if len(s) >= 4:
            lr = stats.linregress(np.arange(len(s)), s.values); growth = 100 * (s.iloc[-3:].mean() / s.iloc[:3].mean() - 1) if s.iloc[:3].mean() else np.nan
            direction = "upward" if lr.slope > 0 else "downward"
            sig = lr.pvalue < .05
            out.append(_i("Trend", "info" if lr.slope > 0 else "high", f"'{metric}' shows a {'significant ' if sig else 'weak '}{direction} trend",
                          f"Monthly slope {human(lr.slope)} (p = {lr.pvalue:.3g}); last 3 months vs first 3: {growth:+.1f}%.",
                          "Sustain drivers of growth." if lr.slope > 0 else "Diagnose the decline by segment and period."))
            z = (s - s.mean()) / (s.std() or 1)
            for ts, zz in z[z.abs() > 2.5].items():
                out.append(_i("Anomaly", "medium", f"Unusual {metric} in {ts:%b %Y} (z = {zz:+.1f})", f"Value {human(s[ts])} vs typical {human(s.mean())}.", "Check promotions, data errors or one-off events."))
            if len(s) >= 24:
                seas = s.groupby(s.index.month).mean(); pk, lo = seas.idxmax(), seas.idxmin()
                out.append(_i("Seasonality", "info", f"Peak month is {pk}, weakest is {lo}", f"Peak is {100 * (seas.max() / seas.min() - 1):.0f}% above the weakest month.", "Align inventory, staffing and campaigns with seasonality."))
    # --- segment opportunities
    if metric and cats:
        for g in cats[:3]:
            agg = df.groupby(g)[metric].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
            if len(agg) < 2: continue
            share = agg["sum"] / agg["sum"].sum()
            out.append(_i("Opportunity", "info", f"'{agg.index[0]}' drives {100 * share.iloc[0]:.0f}% of {metric} within '{g}'", f"Top-2 account for {100 * share.iloc[:2].sum():.0f}%.",
                          "Protect the leader; check concentration risk." if share.iloc[0] > .5 else "Replicate what works in the leader."))
            hi = agg[(agg["mean"] > agg["mean"].median() * 1.1) & (agg["count"] < agg["count"].median())]
            if not hi.empty: out.append(_i("Opportunity", "low", f"High-value, low-volume '{g}' segments: {', '.join(map(str, hi.index[:3]))}", "Above-median average value but below-median volume.", "Grow volume via targeted marketing."))
            groups = [x[metric].dropna().values for _, x in df.groupby(g) if x[metric].notna().sum() > 1]
            if len(groups) >= 2 and len(df) > 30:
                f, p = stats.f_oneway(*groups)
                if p < .05: out.append(_i("Statistical", "info", f"'{metric}' differs significantly across '{g}' (ANOVA p = {p:.3g})", "Group membership explains part of the variation.", f"Tailor strategy per '{g}'."))
        if len(agg) > 4:
            p = df.groupby(cats[0])[metric].sum().sort_values(ascending=False); cum = p.cumsum() / p.sum()
            n80 = int((cum < .8).sum() + 1)
            if n80 / len(p) < .3: out.append(_i("Opportunity", "medium", f"Pareto: {n80}/{len(p)} '{cats[0]}' values generate 80% of {metric}", "Revenue is concentrated.", "Focus retention on the vital few; review the long tail."))
    for c in cats:
        vc = df[c].value_counts(normalize=True)
        if len(vc) > 1 and vc.iloc[0] > .9: out.append(_i("Distribution", "low", f"'{c}' is dominated by '{vc.index[0]}' ({100 * vc.iloc[0]:.0f}%)", "Little variation limits its analytical value.", "Consider dropping or regrouping."))
    # --- multivariate anomalies
    if len(num) >= 2 and len(df) > 50:
        from sklearn.ensemble import IsolationForest
        X = df[num].fillna(df[num].median()); an = IsolationForest(contamination=.01, random_state=0).fit_predict(X) == -1
        out.append(_i("Anomaly", "low", f"{int(an.sum())} multivariate anomalies detected (Isolation Forest, top 1%)", "Records unusual across several columns jointly.", "Review these records for errors or fraud."))
    # --- model
    if ml_result:
        r = ml_result["results"].iloc[0]; col = ml_result["sort_col"]
        out.append(_i("Model", "info", f"Best model for '{ml_result['target']}': {r.Model} ({col} = {r[col]:.3f})", "Evaluated on a held-out test set with leakage-safe preprocessing.", "See Explainable AI for drivers."))
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return sorted(out, key=lambda x: order[x["severity"]])


def render(df):
    st.header("💡 Automated Insights")
    ins = generate(df, st.session_state.get("ml"))
    c1, c2 = st.columns([1, 2])
    kinds = sorted({i["type"] for i in ins})
    sel = c2.multiselect("Filter by type", kinds, default=kinds)
    c1.metric("Insights found", len(ins))
    for i in [x for x in ins if x["type"] in sel]:
        with st.expander(f"{ICON[i['severity']]} [{i['type']}] {i['title']}"):
            st.write(i["detail"])
            if i["action"]: st.markdown(f"**Recommended action:** {i['action']}")
    st.session_state["insights"] = ins
    recs = [i["action"] for i in ins if i["action"] and i["severity"] in ("high", "medium")]
    st.subheader("Top recommendations")
    for r in list(dict.fromkeys(recs))[:6] or ["No urgent issues detected."]: st.markdown(f"- {r}")
