"""Business intelligence: revenue/profit, segmentation, cohort, RFM, Pareto, KPIs, trends."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .utils import detect_types, show, guess_col, idx_of, kpi_row, human, need, cat_like


def _pick(label, options, role, key, allow_none=False, exclude=()):
    opts = ([None] if allow_none else []) + list(options)
    g = guess_col(options, role, exclude)
    return st.selectbox(label, opts, index=idx_of(opts, g), key=key)


def rfm_table(df, cust, date, money, n_bins=5):
    d = df[[cust, date, money]].dropna()
    snap = d[date].max() + pd.Timedelta(days=1)
    r = d.groupby(cust).agg(recency=(date, lambda s: (snap - s.max()).days), frequency=(date, "count"), monetary=(money, "sum"))
    q = min(n_bins, r.shape[0])
    def score(s, asc): return pd.qcut(s.rank(method="first", ascending=asc), q, labels=range(1, q + 1)).astype(int)
    r["R"], r["F"], r["M"] = score(r.recency, False), score(r.frequency, True), score(r.monetary, True)
    r["RFM"] = r.R + r.F + r.M
    def seg(x):
        top = q
        if x.R >= top - 1 and x.F >= top - 1: return "Champions"
        if x.R >= top - 1: return "Recent / Promising"
        if x.F >= top - 1 and x.R <= 2: return "At Risk (loyal)"
        if x.R <= 2 and x.F <= 2: return "Lost / Hibernating"
        return "Needs Attention"
    r["segment"] = r.apply(seg, axis=1)
    return r.reset_index()


def render(df):
    st.header("💼 Business Intelligence")
    types = detect_types(df)
    num, dts, cats = types["numeric"], types["datetime"], cat_like(df, types)
    allc = list(df.columns)
    tabs = st.tabs(["Revenue & Profit", "Trends", "Pareto", "RFM", "Cohort", "Segmentation", "KPI Monitor"])

    with tabs[0]:
        need(num, "Need numeric columns.")
        rev = _pick("Revenue / value column", num, "revenue", "bi_rev")
        cost = _pick("Cost column (optional)", num, "cost", "bi_cost", True, [rev])
        prof = _pick("Profit column (optional)", num, "profit", "bi_prof", True, [rev])
        d = df.copy()
        if prof: d["_profit"] = d[prof]
        elif cost: d["_profit"] = d[rev] - d[cost]
        total = d[rev].sum()
        items = [("Total revenue", human(total), ""), ("Avg per record", human(d[rev].mean()), ""), ("Records", f"{len(d):,}", "")]
        if "_profit" in d:
            items += [("Total profit", human(d["_profit"].sum()), ""), ("Margin", f"{100 * d['_profit'].sum() / total:.1f}%" if total else "n/a", "")]
        kpi_row(items)
        if cats:
            g = st.selectbox("Break down by", cats, key="bi_g")
            agg = d.groupby(g).agg(revenue=(rev, "sum"), **({"profit": ("_profit", "sum")} if "_profit" in d else {})).reset_index()
            if "profit" in agg: agg["margin_%"] = (100 * agg.profit / agg.revenue.replace(0, np.nan)).round(2)
            show(px.bar(agg.sort_values("revenue", ascending=False), x=g, y=[c for c in ["revenue", "profit"] if c in agg], barmode="group"), f"Revenue & profit by {g}")
            st.dataframe(agg.round(2))

    with tabs[1]:
        need(dts and num, "Need a date column and a numeric column.")
        dc = st.selectbox("Date", dts, key="tr_d"); v = _pick("Metric", num, "revenue", "tr_v")
        fr = st.selectbox("Granularity", ["D", "W", "M", "Q"], index=2, key="tr_f"); fn = st.selectbox("Aggregation", ["sum", "mean", "count"], key="tr_a")
        s = df.set_index(dc)[v].resample({"D": "D", "W": "W", "M": "MS", "Q": "QS"}[fr]).agg(fn).dropna()
        need(len(s) >= 3, "Not enough periods.")
        t = pd.DataFrame({"period": s.index, "value": s.values}); t["ma"] = t.value.rolling(3, min_periods=1).mean()
        t["growth_%"] = t.value.pct_change() * 100
        fig = go.Figure(go.Scatter(x=t.period, y=t.value, name=v)); fig.add_scatter(x=t.period, y=t.ma, name="3-period MA")
        show(fig, f"{fn} of {v} over time")
        slope = np.polyfit(np.arange(len(t)), t.value, 1)[0]
        st.write(f"Trend: **{'upward 📈' if slope > 0 else 'downward 📉'}** ({slope:,.2f} per period). Avg growth: **{t['growth_%'].mean():.2f}%**")
        show(px.bar(t, x="period", y="growth_%"), "Period-over-period growth %")
        if len(t) >= 24 or (fr == "M" and len(t) >= 12):
            t["month"] = t.period.dt.month
            show(px.bar(t.groupby("month").value.mean().reset_index(), x="month", y="value"), "Seasonality – average by month")

    with tabs[2]:
        need(num and allc, "Need a numeric column.")
        v = _pick("Value", num, "revenue", "pa_v"); g = st.selectbox("Item / category", [c for c in allc if c != v], key="pa_g")
        p = df.groupby(g)[v].sum().sort_values(ascending=False).reset_index()
        p["cum_%"] = 100 * p[v].cumsum() / p[v].sum()
        top = p.head(st.slider("Show top N", 5, min(100, max(6, len(p))), min(20, max(6, len(p)))))
        fig = go.Figure(go.Bar(x=top[g].astype(str), y=top[v], name=v))
        fig.add_scatter(x=top[g].astype(str), y=top["cum_%"], name="Cumulative %", yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105]))
        show(fig, f"Pareto analysis – {v} by {g}")
        n80 = int((p["cum_%"] < 80).sum() + 1)
        st.success(f"**{n80} of {len(p)} {g} values ({100 * n80 / len(p):.1f}%)** generate ~80% of total {v}.")

    with tabs[3]:
        need(dts and num and len(allc) > 2, "RFM needs a customer column, a date column and a monetary column.")
        cust = _pick("Customer ID", allc, "customer", "rfm_c"); dc = st.selectbox("Transaction date", dts, key="rfm_d")
        mon = _pick("Monetary value", num, "revenue", "rfm_m")
        if df[cust].nunique() < 5: st.warning("Too few unique customers for RFM."); st.stop()
        r = rfm_table(df, cust, dc, mon)
        st.session_state["rfm"] = r
        s = r.groupby("segment").agg(customers=(cust, "count"), avg_recency=("recency", "mean"), avg_freq=("frequency", "mean"), revenue=("monetary", "sum")).reset_index()
        a, b = st.columns(2)
        with a: show(px.pie(s, names="segment", values="customers", hole=.45), "Customers by RFM segment")
        with b: show(px.bar(s, x="segment", y="revenue", color="segment"), "Revenue by RFM segment")
        st.dataframe(s.round(2)); show(px.scatter(r, x="frequency", y="monetary", color="segment", size="monetary", hover_name=cust), "Frequency vs monetary")
        st.download_button("Download RFM table", r.to_csv(index=False).encode(), "rfm.csv")

    with tabs[4]:
        need(dts and len(allc) > 1, "Cohort analysis needs a customer column and a date column.")
        cust = _pick("Customer ID", allc, "customer", "co_c"); dc = st.selectbox("Date", dts, key="co_d")
        d = df[[cust, dc]].dropna().copy(); d["order_m"] = d[dc].dt.to_period("M")
        d["cohort"] = d.groupby(cust)["order_m"].transform("min")
        d["idx"] = (d.order_m - d.cohort).apply(lambda x: x.n)
        c = d.groupby(["cohort", "idx"])[cust].nunique().unstack(fill_value=0)
        need(c.shape[0] > 1 and c.shape[1] > 1, "Not enough periods for cohort analysis.")
        ret = (c.div(c[0], axis=0) * 100).round(1); ret.index = ret.index.astype(str)
        show(px.imshow(ret, text_auto=".0f", aspect="auto", color_continuous_scale="Blues", labels=dict(x="Months since first purchase", y="Cohort", color="Retention %")), "Cohort retention (%)", height=520)
        st.write(f"Average retention in month 1: **{ret[1].mean():.1f}%**" if 1 in ret.columns else "")

    with tabs[5]:
        need(len(num) >= 2, "Need at least two numeric columns.")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        feats = st.multiselect("Features", num, default=num[:min(4, len(num))], key="seg_f")
        need(len(feats) >= 2, "Select at least two features.")
        X = StandardScaler().fit_transform(df[feats].fillna(df[feats].median()))
        samp = np.random.default_rng(0).choice(len(X), min(len(X), 3000), replace=False)
        res = []
        for k in range(2, 9):
            km = KMeans(k, n_init=5, random_state=42).fit(X); res.append((k, km.inertia_, silhouette_score(X[samp], km.labels_[samp])))
        res = pd.DataFrame(res, columns=["k", "inertia", "silhouette"]); best = int(res.loc[res.silhouette.idxmax(), "k"])
        a, b = st.columns(2)
        with a: show(px.line(res, x="k", y="inertia", markers=True), "Elbow curve", height=300)
        with b: show(px.line(res, x="k", y="silhouette", markers=True), "Silhouette score", height=300)
        k = st.slider("Number of segments", 2, 8, best); st.caption(f"Best silhouette at k = {best}")
        lab = KMeans(k, n_init=10, random_state=42).fit_predict(X)
        seg = df.copy(); seg["segment"] = lab.astype(str); st.session_state["segments"] = seg
        from sklearn.decomposition import PCA
        pc = PCA(2).fit_transform(X); pdf = pd.DataFrame(pc, columns=["PC1", "PC2"]); pdf["segment"] = lab.astype(str)
        show(px.scatter(pdf.iloc[samp], x="PC1", y="PC2", color="segment", opacity=.6), "Segments (PCA projection)")
        prof = seg.groupby("segment")[feats].mean(); prof["size"] = seg.segment.value_counts(); st.dataframe(prof.round(2))

    with tabs[6]:
        need(dts and num, "KPI monitoring needs a date and numeric columns.")
        dc = st.selectbox("Date", dts, key="k_d"); ks = st.multiselect("KPIs (sum per period)", num, default=num[:3], key="k_k")
        fr = st.selectbox("Period", ["M", "W", "Q"], key="k_f"); tol = st.slider("Alert threshold ±%", 1, 50, 10)
        rows = []
        for k in ks:
            s = df.set_index(dc)[k].resample({"M": "MS", "W": "W", "Q": "QS"}[fr]).sum()
            if len(s) >= 2 and s.iloc[-2] != 0:
                ch = 100 * (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]); rows.append((k, s.iloc[-1], s.iloc[-2], ch))
        if rows:
            cols = st.columns(len(rows))
            for col, (k, cur, prev, ch) in zip(cols, rows):
                col.metric(k, human(cur), f"{ch:+.1f}%")
                (col.error if ch < -tol else col.success if ch > tol else col.info)("ALERT" if abs(ch) > tol else "On track")
            st.caption("Note: the latest period may be partial.")
