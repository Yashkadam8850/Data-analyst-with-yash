"""Executive dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .utils import detect_types, quality_report, kpi_row, show, human, memory_mb, guess_col, cat_like


def render(df):
    st.header("📊 Executive Dashboard")
    types = detect_types(df); q = quality_report(df, types)
    color = "#16a34a" if q["score"] >= 90 else "#f59e0b" if q["score"] >= 75 else "#dc2626"
    kpi_row([("Total records", f"{len(df):,}", ""), ("Columns", f"{df.shape[1]}", f"{len(types['numeric'])} num · {len(types['categorical'])} cat · {len(types['datetime'])} date"),
             ("Missing values", f"{q['missing']:,}", f"{q['missing_pct']:.2f}% of cells"), ("Duplicate rows", f"{q['duplicates']:,}", ""),
             ("Data quality", f"{q['score']}/100", f"memory {memory_mb(df):.1f} MB")])
    st.session_state["quality"] = q
    num, dts, cats = types["numeric"], types["datetime"], cat_like(df, types)
    if num:
        pref = [c for c in [guess_col(num, r) for r in ("revenue", "profit", "quantity", "cost")] if c]
        kc = list(dict.fromkeys(pref + num))[:4]
        st.subheader("Key metrics")
        kpi_row([(f"Σ {c}", human(df[c].sum()), f"avg {human(df[c].mean())} · median {human(df[c].median())}") for c in kc])
    a, b = st.columns(2)
    with a:
        tc = pd.Series({"Numeric": len(types["numeric"]), "Categorical": len(types["categorical"]), "Datetime": len(types["datetime"]), "ID-like": len(types["id"])})
        show(px.pie(tc[tc > 0].reset_index(), names="index", values=0, hole=.5), "Column types", height=340)
    with b:
        comp = pd.DataFrame({"dimension": ["Completeness", "Uniqueness", "Consistency", "Validity"], "score": [100 * q[k] for k in ("completeness", "uniqueness", "consistency", "validity")]})
        show(px.bar(comp, x="score", y="dimension", orientation="h", range_x=[0, 100]), "Data-quality dimensions", height=340)
    if dts and num:
        d = dts[0]; v = guess_col(num, "revenue") or num[0]
        s = df.set_index(d)[v].resample("MS").sum().reset_index()
        if len(s) > 2: show(px.area(s, x=d, y=v), f"Monthly {v}", height=360)
    a, b = st.columns(2)
    with a:
        if cats and num:
            g = cats[0]; v = guess_col(num, "revenue") or num[0]
            t = df.groupby(g)[v].sum().sort_values(ascending=False).head(10).reset_index()
            show(px.bar(t, x=g, y=v, color=g), f"{v} by {g}", height=360)
    with b:
        if len(num) >= 2:
            c = df[num].corr(); show(px.imshow(c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), "Correlation heatmap", height=360)
    miss = df.isna().sum(); miss = miss[miss > 0]
    if len(miss): show(px.bar(miss.reset_index(), x="index", y=0, labels={"index": "column", "0": "missing"}), "Missing values by column", height=320)
    st.subheader("Data preview"); st.dataframe(df.head(50))
