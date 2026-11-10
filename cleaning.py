"""Advanced data cleaning: missing values, duplicates, outliers, types, feature engineering."""
import numpy as np
import pandas as pd
import streamlit as st

from .utils import detect_types, outlier_mask, show, kpi_row, quality_report
import plotly.express as px


def _log(msg):
    st.session_state.setdefault("clean_log", []).append(msg)


def render(df_raw, df):
    st.header("🧹 Advanced Data Cleaning")
    types = detect_types(df)
    q = quality_report(df, types)
    kpi_row([("Rows", f"{len(df):,}", ""), ("Missing cells", f"{q['missing']:,}", f"{q['missing_pct']:.2f}%"),
             ("Duplicates", f"{q['duplicates']:,}", ""), ("Quality score", f"{q['score']}/100", "")])
    t1, t2, t3, t4, t5, t6 = st.tabs(["Missing values", "Duplicates", "Outliers", "Data types", "Feature engineering", "Log / reset"])

    # ---- missing values
    with t1:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            st.success("No missing values.")
        else:
            mdf = pd.DataFrame({"missing": miss, "pct": (100 * miss / len(df)).round(2)})
            st.dataframe(mdf)
            show(px.bar(mdf.reset_index(), x="index", y="pct", labels={"index": "column", "pct": "% missing"}), "Missing % by column")
            c1, c2, c3 = st.columns(3)
            cols = c1.multiselect("Columns", list(miss.index), default=list(miss.index))
            num_s = c2.selectbox("Numeric strategy", ["median", "mean", "mode", "constant 0", "interpolate", "KNN", "drop rows"])
            cat_s = c3.selectbox("Categorical strategy", ["mode", "'Unknown'", "drop rows"])
            thr = st.slider("Drop columns with missing % above", 0, 100, 100)
            if st.button("Apply missing-value treatment", type="primary"):
                d = df.copy()
                drop_cols = [c for c in d.columns if 100 * d[c].isna().mean() > thr]
                if thr < 100 and drop_cols:
                    d = d.drop(columns=drop_cols); _log(f"Dropped columns >{thr}% missing: {drop_cols}")
                cols = [c for c in cols if c in d.columns]
                ncols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
                ccols = [c for c in cols if c not in ncols]
                if num_s == "KNN" and ncols:
                    from sklearn.impute import KNNImputer
                    allnum = d.select_dtypes("number").columns
                    d[allnum] = KNNImputer(n_neighbors=5).fit_transform(d[allnum])
                else:
                    for c in ncols:
                        if num_s == "median": d[c] = d[c].fillna(d[c].median())
                        elif num_s == "mean": d[c] = d[c].fillna(d[c].mean())
                        elif num_s == "mode": d[c] = d[c].fillna(d[c].mode().iloc[0] if not d[c].mode().empty else 0)
                        elif num_s == "constant 0": d[c] = d[c].fillna(0)
                        elif num_s == "interpolate": d[c] = d[c].interpolate(limit_direction="both")
                        elif num_s == "drop rows": d = d.dropna(subset=[c])
                for c in ccols:
                    if pd.api.types.is_datetime64_any_dtype(d[c]):
                        d[c] = d[c].ffill().bfill()
                    elif cat_s == "mode":
                        m = d[c].mode()
                        d[c] = d[c].fillna(m.iloc[0] if not m.empty else "Unknown")
                    elif cat_s == "'Unknown'":
                        d[c] = d[c].astype("object").fillna("Unknown")
                    else:
                        d = d.dropna(subset=[c])
                st.session_state.df_clean = d
                _log(f"Missing values: numeric={num_s}, categorical={cat_s} on {len(cols)} columns")
                st.rerun()

    # ---- duplicates
    with t2:
        subset = st.multiselect("Consider columns (empty = all)", list(df.columns))
        dm = df.duplicated(subset=subset or None, keep=False)
        st.metric("Rows involved in duplication", int(dm.sum()))
        if dm.any():
            st.dataframe(df[dm].sort_values(subset or list(df.columns)[:1]).head(200))
        keep = st.radio("Keep", ["first", "last"], horizontal=True)
        if st.button("Remove duplicates") and dm.any():
            n0 = len(df)
            st.session_state.df_clean = df.drop_duplicates(subset=subset or None, keep=keep).reset_index(drop=True)
            _log(f"Removed {n0 - len(st.session_state.df_clean)} duplicate rows"); st.rerun()

    # ---- outliers
    with t3:
        nums = types["numeric"]
        if not nums:
            st.info("No numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            method = c1.selectbox("Method", ["IQR", "Z-score", "Isolation Forest"])
            thr = c2.number_input("Threshold (k / z)", 0.5, 10.0, 1.5 if method == "IQR" else 3.0, 0.5)
            cols = c3.multiselect("Columns", nums, default=nums[:min(3, len(nums))])
            if cols:
                if method == "Isolation Forest":
                    from sklearn.ensemble import IsolationForest
                    X = df[cols].fillna(df[cols].median())
                    mask = pd.Series(IsolationForest(contamination=0.02, random_state=42).fit_predict(X) == -1, index=df.index)
                    counts = {"(multivariate)": int(mask.sum())}
                else:
                    masks = {c: outlier_mask(df[c], method, thr) for c in cols}
                    counts = {c: int(m.sum()) for c, m in masks.items()}
                    mask = pd.concat(masks.values(), axis=1).any(axis=1)
                st.write("Outliers per column:", counts)
                if method != "Isolation Forest":
                    show(px.box(df, y=cols[0], points="outliers"), f"Box plot – {cols[0]}")
                action = st.selectbox("Action", ["Flag only (add column)", "Cap (winsorize)", "Remove rows"])
                if st.button("Apply outlier treatment"):
                    d = df.copy()
                    if action.startswith("Flag"):
                        d["is_outlier"] = mask.astype(int)
                    elif action.startswith("Cap") and method != "Isolation Forest":
                        for c in cols:
                            lo, hi = d[c].quantile([.01, .99]) if method == "Z-score" else (
                                d[c].quantile(.25) - thr * (d[c].quantile(.75) - d[c].quantile(.25)),
                                d[c].quantile(.75) + thr * (d[c].quantile(.75) - d[c].quantile(.25)))
                            d[c] = d[c].clip(lo, hi)
                    else:
                        d = d[~mask]
                    st.session_state.df_clean = d.reset_index(drop=True)
                    _log(f"Outliers ({method}, {thr}) – {action} on {cols}"); st.rerun()

    # ---- types
    with t4:
        tdf = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values, "unique": df.nunique().values})
        st.dataframe(tdf)
        c1, c2 = st.columns(2)
        col = c1.selectbox("Column to convert", list(df.columns))
        to = c2.selectbox("Convert to", ["numeric", "datetime", "category", "string", "integer"])
        if st.button("Convert"):
            d = df.copy()
            try:
                if to == "numeric": d[col] = pd.to_numeric(d[col].astype(str).str.replace(r"[,$€£%\s]", "", regex=True), errors="coerce")
                elif to == "datetime": d[col] = pd.to_datetime(d[col], errors="coerce")
                elif to == "category": d[col] = d[col].astype("category")
                elif to == "string": d[col] = d[col].astype(str)
                else: d[col] = pd.to_numeric(d[col], errors="coerce").round().astype("Int64")
                bad = int(d[col].isna().sum() - df[col].isna().sum())
                st.session_state.df_clean = d
                _log(f"Converted {col} -> {to} ({bad} values became NaN)"); st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    # ---- feature engineering
    with t5:
        op = st.selectbox("Operation", ["Date parts", "Log transform", "Binning", "Ratio of two columns",
                                         "Product of two columns", "Standardize (z-score)", "One-hot encode", "Rolling mean"])
        d = df.copy(); ok = False
        if op == "Date parts" and types["datetime"]:
            c = st.selectbox("Date column", types["datetime"])
            if st.button("Create"):
                p = d[c].dt
                d[f"{c}_year"], d[f"{c}_month"], d[f"{c}_quarter"] = p.year, p.month, p.quarter
                d[f"{c}_dayofweek"], d[f"{c}_is_weekend"] = p.dayofweek, (p.dayofweek >= 5).astype(int); ok = True
        elif op in ("Log transform", "Standardize (z-score)", "Binning", "Rolling mean") and types["numeric"]:
            c = st.selectbox("Numeric column", types["numeric"])
            if op == "Binning":
                nb = st.slider("Bins", 2, 20, 5)
            if op == "Rolling mean":
                w = st.slider("Window", 2, 60, 7)
            if st.button("Create"):
                if op == "Log transform": d[f"{c}_log"] = np.log1p(d[c].clip(lower=0)); ok = True
                elif op == "Standardize (z-score)": d[f"{c}_z"] = (d[c] - d[c].mean()) / (d[c].std() or 1); ok = True
                elif op == "Binning": d[f"{c}_bin"] = pd.qcut(d[c], nb, duplicates="drop").astype(str); ok = True
                else: d[f"{c}_roll{w}"] = d[c].rolling(w, min_periods=1).mean(); ok = True
        elif op in ("Ratio of two columns", "Product of two columns") and len(types["numeric"]) >= 2:
            a = st.selectbox("Column A", types["numeric"]); b = st.selectbox("Column B", types["numeric"], index=1)
            if st.button("Create"):
                if op.startswith("Ratio"): d[f"{a}_per_{b}"] = d[a] / d[b].replace(0, np.nan)
                else: d[f"{a}_x_{b}"] = d[a] * d[b]
                ok = True
        elif op == "One-hot encode" and types["categorical"]:
            c = st.multiselect("Categorical columns", types["categorical"])
            if st.button("Create") and c:
                d = pd.get_dummies(d, columns=c, drop_first=True, dtype=int); ok = True
        else:
            st.info("No suitable columns for this operation.")
        if ok:
            st.session_state.df_clean = d; _log(f"Feature engineering: {op}"); st.rerun()

    # ---- log
    with t6:
        for m in st.session_state.get("clean_log", []) or ["No cleaning steps applied yet."]:
            st.write("•", m)
        if st.button("↩️ Reset to original data"):
            st.session_state.df_clean = df_raw.copy(); st.session_state.clean_log = []; st.rerun()
    st.subheader("Preview")
    st.dataframe(df.head(100))
