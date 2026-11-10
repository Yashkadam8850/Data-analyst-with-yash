"""Exploratory data analysis."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from .utils import detect_types, show, cat_like, need


def render(df):
    st.header("🔎 Exploratory Data Analysis")
    types = detect_types(df)
    num, cats = types["numeric"], cat_like(df, types)
    t1, t2, t3, t4, t5 = st.tabs(["Summary", "Univariate", "Bivariate", "Multivariate", "Distributions"])

    with t1:
        if num:
            d = df[num].describe().T
            d["skew"], d["kurtosis"] = df[num].skew(), df[num].kurt()
            d["missing"] = df[num].isna().sum()
            st.dataframe(d.round(3))
        oc = [c for c in df.columns if c not in num]
        if oc:
            st.write("**Non-numeric columns**")
            st.dataframe(df[oc].describe(include="all").T.astype(str))

    with t2:
        c = st.selectbox("Column", list(df.columns), key="uni")
        if c in num:
            a, b = st.columns(2)
            with a: show(px.histogram(df, x=c, nbins=40, marginal="box"), f"Distribution – {c}")
            with b:
                s = df[c].dropna()
                st.dataframe(pd.DataFrame({"value": [s.mean(), s.median(), s.std(), s.skew(), s.kurt(), s.min(), s.max(),
                                                      s.quantile(.25), s.quantile(.75)]},
                             index=["mean", "median", "std", "skew", "kurtosis", "min", "max", "Q1", "Q3"]).round(4))
        elif c in types["datetime"]:
            v = df[c].dt.to_period("M").value_counts().sort_index()
            show(px.bar(x=v.index.astype(str), y=v.values, labels={"x": c, "y": "count"}), f"Records per month – {c}")
        else:
            v = df[c].value_counts().head(25).reset_index()
            v.columns = [c, "count"]
            show(px.bar(v, x=c, y="count"), f"Top categories – {c}")

    with t3:
        mode = st.radio("Pair type", ["Numeric vs Numeric", "Category vs Numeric", "Category vs Category"], horizontal=True)
        if mode == "Numeric vs Numeric" and len(num) >= 2:
            x = st.selectbox("X", num, key="bx"); y = st.selectbox("Y", num, index=1, key="by")
            color = st.selectbox("Color by", [None] + cats, key="bc")
            d = df.sample(min(len(df), 5000), random_state=1)
            try: fig = px.scatter(d, x=x, y=y, color=color, trendline="ols" if color is None else None, opacity=.6)
            except Exception: fig = px.scatter(d, x=x, y=y, color=color, opacity=.6)
            show(fig, f"{y} vs {x}")
            ok = df[[x, y]].dropna()
            if len(ok) > 3:
                r, p = stats.pearsonr(ok[x], ok[y]); rho, p2 = stats.spearmanr(ok[x], ok[y])
                st.write(f"Pearson r = **{r:.3f}** (p={p:.2g}) · Spearman ρ = **{rho:.3f}** (p={p2:.2g})")
        elif mode == "Category vs Numeric" and cats and num:
            c = st.selectbox("Category", cats, key="cn1"); n = st.selectbox("Numeric", num, key="cn2")
            show(px.box(df, x=c, y=n, color=c), f"{n} by {c}")
            st.dataframe(df.groupby(c)[n].agg(["count", "mean", "median", "std"]).round(3))
        elif mode == "Category vs Category" and len(cats) >= 2:
            a = st.selectbox("Category A", cats, key="cc1"); b = st.selectbox("Category B", cats, index=1, key="cc2")
            ct = pd.crosstab(df[a], df[b])
            show(px.imshow(ct, text_auto=True, aspect="auto"), f"Contingency – {a} × {b}")
        else:
            st.info("Not enough suitable columns for this analysis.")

    with t4:
        need(len(num) >= 2, "Need at least two numeric columns.")
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        corr = df[num].corr(method=method)
        show(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
             f"Correlation matrix ({method})", height=520)
        pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().reset_index()
        pairs.columns = ["A", "B", "r"]
        st.write("**Strongest relationships**")
        st.dataframe(pairs.reindex(pairs.r.abs().sort_values(ascending=False).index).head(10).round(3))
        if len(num) >= 3:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(df[num].fillna(df[num].median()))
            pca = PCA().fit(X)
            ev = pca.explained_variance_ratio_
            show(px.bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, labels={"x": "", "y": "explained variance"}), "PCA explained variance")
        sel = st.multiselect("Scatter matrix columns", num, default=num[:4])
        if len(sel) >= 2:
            show(px.scatter_matrix(df.sample(min(len(df), 2000), random_state=1), dimensions=sel, color=cats[0] if cats else None),
                 "Scatter matrix", height=600)

    with t5:
        need(num, "No numeric columns.")
        rows = []
        for c in num:
            s = df[c].dropna()
            if len(s) < 8: continue
            samp = s.sample(min(len(s), 5000), random_state=1)
            p = stats.shapiro(samp)[1] if len(samp) >= 3 else np.nan
            rows.append(dict(column=c, skew=s.skew(), kurtosis=s.kurt(), shapiro_p=p, normal=p > .05,
                             shape="right-skewed" if s.skew() > 1 else "left-skewed" if s.skew() < -1 else "approx. symmetric"))
        st.dataframe(pd.DataFrame(rows).round(4))
        c = st.selectbox("Column", num, key="dist")
        s = df[c].dropna()
        fig = go.Figure(go.Histogram(x=s, histnorm="probability density", name="data", nbinsx=40))
        xs = np.linspace(s.min(), s.max(), 200)
        fig.add_scatter(x=xs, y=stats.norm.pdf(xs, s.mean(), s.std() or 1), name="normal fit")
        show(fig, f"Histogram vs normal curve – {c}")
        qq = stats.probplot(s, dist="norm")
        f2 = go.Figure(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="sample"))
        f2.add_scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * qq[0][0], name="reference")
        show(f2, f"Q-Q plot – {c}")
