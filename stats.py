"""Advanced statistical analysis: hypothesis tests, correlation tests, regression diagnostics, CIs."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from .utils import detect_types, show, cat_like, need


def _verdict(p, alpha):
    return (f"✅ **Reject H₀** (p = {p:.4g} < α = {alpha}) – statistically significant."
            if p < alpha else f"❌ **Fail to reject H₀** (p = {p:.4g} ≥ α = {alpha}) – no significant evidence.")


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp else np.nan


def render(df):
    st.header("📐 Advanced Statistical Analysis")
    types = detect_types(df)
    num, cats = types["numeric"], cat_like(df, types)
    alpha = st.sidebar.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
    tabs = st.tabs(["T-test", "ANOVA", "Chi-square", "Correlation", "Regression diagnostics", "Confidence intervals"])

    with tabs[0]:
        need(num, "Need a numeric column.")
        kind = st.radio("Test", ["Independent (Welch)", "One-sample", "Paired"], horizontal=True)
        v = st.selectbox("Numeric variable", num, key="t_v")
        if kind.startswith("Independent"):
            need(cats, "Need a grouping column.")
            g = st.selectbox("Group column", cats, key="t_g")
            lv = df[g].dropna().unique().tolist()
            need(len(lv) >= 2, "Group column needs ≥ 2 levels.")
            a, b = st.columns(2)
            g1 = a.selectbox("Group 1", lv, key="t_1"); g2 = b.selectbox("Group 2", lv, index=1, key="t_2")
            x, y = df.loc[df[g] == g1, v].dropna(), df.loc[df[g] == g2, v].dropna()
            need(len(x) > 2 and len(y) > 2, "Each group needs more than 2 observations.")
            t, p = stats.ttest_ind(x, y, equal_var=False)
            st.write(f"t = **{t:.4f}**, p = **{p:.4g}**, Cohen's d = **{cohens_d(x, y):.3f}**, "
                     f"means: {x.mean():.3f} vs {y.mean():.3f}")
            st.write(f"Levene test (equal variance) p = {stats.levene(x, y)[1]:.4g} · Mann-Whitney U p = {stats.mannwhitneyu(x, y)[1]:.4g}")
            st.markdown(_verdict(p, alpha))
            show(px.box(df[df[g].isin([g1, g2])], x=g, y=v, color=g), f"{v}: {g1} vs {g2}")
        elif kind == "One-sample":
            mu = st.number_input("Hypothesised mean μ₀", value=float(df[v].mean()))
            s = df[v].dropna(); t, p = stats.ttest_1samp(s, mu)
            st.write(f"t = **{t:.4f}**, p = **{p:.4g}**, sample mean = {s.mean():.4f}"); st.markdown(_verdict(p, alpha))
        else:
            need(len(num) >= 2, "Need two numeric columns.")
            v2 = st.selectbox("Second variable", [c for c in num if c != v], key="t_p2")
            d = df[[v, v2]].dropna(); t, p = stats.ttest_rel(d[v], d[v2])
            st.write(f"t = **{t:.4f}**, p = **{p:.4g}**, mean diff = {(d[v] - d[v2]).mean():.4f}"); st.markdown(_verdict(p, alpha))

    with tabs[1]:
        need(num and cats, "Need a numeric and a categorical column.")
        v = st.selectbox("Numeric variable", num, key="a_v"); g = st.selectbox("Factor", cats, key="a_g")
        groups = [s[v].dropna().values for _, s in df.groupby(g) if s[v].notna().sum() > 1]
        need(len(groups) >= 2, "Need at least two groups with data.")
        f, p = stats.f_oneway(*groups); h, p_kw = stats.kruskal(*groups)
        ss_b = sum(len(x) * (x.mean() - df[v].mean()) ** 2 for x in groups); ss_t = ((df[v].dropna() - df[v].mean()) ** 2).sum()
        st.write(f"F = **{f:.4f}**, p = **{p:.4g}**, η² = **{ss_b / ss_t:.3f}** · Kruskal-Wallis p = {p_kw:.4g}")
        st.markdown(_verdict(p, alpha))
        if p < alpha:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            d = df[[v, g]].dropna()
            tk = pairwise_tukeyhsd(d[v], d[g].astype(str), alpha=alpha)
            st.write("**Tukey HSD post-hoc**")
            st.dataframe(pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0]))
        show(px.box(df, x=g, y=v, color=g), f"{v} by {g}")

    with tabs[2]:
        need(len(cats) >= 2, "Need two categorical columns.")
        a = st.selectbox("Variable A", cats, key="c_a"); b = st.selectbox("Variable B", [c for c in cats if c != a], key="c_b")
        ct = pd.crosstab(df[a], df[b]); need(ct.shape[0] > 1 and ct.shape[1] > 1, "Each variable needs ≥ 2 levels.")
        chi, p, dof, exp = stats.chi2_contingency(ct)
        v = np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1)))
        st.write(f"χ² = **{chi:.4f}**, dof = {dof}, p = **{p:.4g}**, Cramér's V = **{v:.3f}**")
        if (exp < 5).mean() > .2: st.warning("More than 20% of expected counts < 5 – interpret with caution.")
        st.markdown(_verdict(p, alpha)); st.dataframe(ct)
        show(px.imshow(ct, text_auto=True, aspect="auto"), f"Observed counts – {a} × {b}")

    with tabs[3]:
        need(len(num) >= 2, "Need two numeric columns.")
        rows = []
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                d = df[[a, b]].dropna()
                if len(d) < 8 or d[a].nunique() < 2 or d[b].nunique() < 2: continue
                r, p = stats.pearsonr(d[a], d[b]); rho, p2 = stats.spearmanr(d[a], d[b]); tau, p3 = stats.kendalltau(d[a], d[b])
                rows.append(dict(A=a, B=b, pearson=r, pearson_p=p, spearman=rho, spearman_p=p2, kendall=tau, kendall_p=p3,
                                 significant=p < alpha))
        res = pd.DataFrame(rows)
        need(not res.empty, "No valid column pairs.")
        st.dataframe(res.reindex(res.pearson.abs().sort_values(ascending=False).index).round(4))

    with tabs[4]:
        need(len(num) >= 2, "Need at least two numeric columns.")
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson, jarque_bera
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        y_col = st.selectbox("Dependent variable", num, key="r_y")
        xs = st.multiselect("Predictors", [c for c in num if c != y_col], default=[c for c in num if c != y_col][:3])
        need(xs, "Select at least one predictor.")
        d = df[[y_col] + xs].dropna(); need(len(d) > len(xs) + 5, "Too few complete rows.")
        X = sm.add_constant(d[xs]); m = sm.OLS(d[y_col], X).fit()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{m.rsquared:.4f}"); c2.metric("Adj. R²", f"{m.rsquared_adj:.4f}")
        c3.metric("F p-value", f"{m.f_pvalue:.3g}"); c4.metric("Durbin-Watson", f"{durbin_watson(m.resid):.2f}")
        co = pd.concat([m.params, m.bse, m.tvalues, m.pvalues, m.conf_int()], axis=1)
        co.columns = ["coef", "std err", "t", "p", "CI low", "CI high"]; st.dataframe(co.round(4))
        bp = het_breuschpagan(m.resid, X)[1]; jb = jarque_bera(m.resid)[1]
        st.write(f"Breusch-Pagan (heteroscedasticity) p = **{bp:.4g}** {'⚠️' if bp < alpha else '✅'} · "
                 f"Jarque-Bera (residual normality) p = **{jb:.4g}** {'⚠️' if jb < alpha else '✅'}")
        if len(xs) > 1:
            vif = pd.DataFrame({"feature": xs, "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(xs))]})
            st.write("**Multicollinearity (VIF > 10 is problematic)**"); st.dataframe(vif.round(3))
        fit = m.fittedvalues
        a, b = st.columns(2)
        with a: show(px.scatter(x=fit, y=m.resid, labels={"x": "Fitted", "y": "Residual"}, opacity=.5), "Residuals vs fitted")
        with b:
            qq = stats.probplot(m.resid, dist="norm")
            f = go.Figure(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers")); f.add_scatter(x=qq[0][0], y=qq[1][1] + qq[1][0] * qq[0][0])
            show(f, "Residual Q-Q plot")
        cook = m.get_influence().cooks_distance[0]
        st.write(f"Influential observations (Cook's D > 4/n): **{int((cook > 4 / len(d)).sum())}**")

    with tabs[5]:
        need(num, "No numeric columns.")
        v = st.selectbox("Variable", num, key="ci_v"); lvl = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        s = df[v].dropna(); n = len(s); m, se = s.mean(), stats.sem(s)
        lo, hi = stats.t.interval(lvl, n - 1, loc=m, scale=se)
        rng = np.random.default_rng(0); boots = [rng.choice(s.values, n).mean() for _ in range(1000)]
        blo, bhi = np.quantile(boots, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
        st.write(f"Mean = **{m:.4f}** · t-interval: **[{lo:.4f}, {hi:.4f}]** · bootstrap: **[{blo:.4f}, {bhi:.4f}]** (n = {n})")
        if cats:
            g = st.selectbox("Compare by group", [None] + cats, key="ci_g")
            if g:
                rows = []
                for k, sub in df.groupby(g):
                    x = sub[v].dropna()
                    if len(x) > 2:
                        l, h = stats.t.interval(lvl, len(x) - 1, loc=x.mean(), scale=stats.sem(x)); rows.append((str(k), x.mean(), l, h))
                r = pd.DataFrame(rows, columns=["group", "mean", "lo", "hi"])
                f = go.Figure(go.Scatter(x=r.group, y=r["mean"], mode="markers",
                              error_y=dict(type="data", symmetric=False, array=r.hi - r["mean"], arrayminus=r["mean"] - r.lo)))
                show(f, f"{int(lvl*100)}% CI of mean {v} by {g}")
