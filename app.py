"""IntelliAnalytics – Advanced Data Intelligence Platform (Streamlit entry point)."""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IntelliAnalytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

from modules import utils  # noqa: E402
from modules import (bi, cleaning, dashboard, eda, explain, insights, ml, predictive, report, stats)  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {"Sales transactions (BI / forecasting)": "sales_transactions.csv", "Customer churn (ML / SHAP)": "customer_churn.csv"}
PAGES = {
    "📊 Executive Dashboard": lambda d, r: dashboard.render(d),
    "🧹 Data Cleaning": lambda d, r: cleaning.render(r, d),
    "🔎 Exploratory Analysis": lambda d, r: eda.render(d),
    "📐 Statistical Analysis": lambda d, r: stats.render(d),
    "💼 Business Intelligence": lambda d, r: bi.render(d),
    "🤖 Machine Learning": lambda d, r: ml.render(d),
    "🔮 Predictive Analytics": lambda d, r: predictive.render(d),
    "🧠 Explainable AI": lambda d, r: explain.render(d),
    "💡 Automated Insights": lambda d, r: insights.render(d),
    "📄 Reporting & Export": lambda d, r: report.render(d, r),
}


def load_dataset():
    st.sidebar.subheader("📁 Data")
    src = st.sidebar.radio("Source", ["Sample dataset", "Upload file"], horizontal=True, label_visibility="collapsed")
    try:
        if src == "Sample dataset":
            name = st.sidebar.selectbox("Sample", list(SAMPLES))
            key, loader = f"sample:{name}", lambda: utils.load_file(DATA_DIR / SAMPLES[name])
        else:
            f = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls", "xlsm"])
            if f is None:
                return False
            sheet = None
            if f.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
                sheets = utils.excel_sheets(f); sheet = st.sidebar.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
            key, loader = f"upload:{f.name}:{f.size}:{sheet}", lambda: utils.load_file(f, sheet)
        if st.session_state.get("data_key") != key:
            with st.spinner("Loading and profiling data..."):
                raw = loader()
            if raw.empty:
                st.sidebar.error("The file contains no data."); return False
            for k in ("ml", "risk", "forecast", "figs", "clean_log", "pdf", "scored", "rfm", "segments"):
                st.session_state.pop(k, None)
            st.session_state.update(data_key=key, df_raw=raw, df_clean=raw.copy())
        return True
    except Exception as e:
        st.sidebar.error(f"Could not load data: {e}")
        return False


def main():
    st.session_state.setdefault("theme", "Light")
    with st.sidebar:
        st.markdown("## 📈 IntelliAnalytics")
        st.caption("Advanced Data Intelligence Platform")
        st.radio("Theme", ["Light", "Dark"], key="theme", horizontal=True)
    utils.apply_theme_css()
    ok = load_dataset()
    if not ok:
        st.title("📈 IntelliAnalytics")
        st.markdown("### Advanced Data Intelligence Platform\nUpload a **CSV or Excel** file in the sidebar (or pick a sample dataset) to begin. "
                    "Column types are detected automatically – no configuration required.")
        return
    raw, df = st.session_state.df_raw, st.session_state.df_clean
    st.sidebar.success(f"{len(df):,} rows × {df.shape[1]} columns")
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
    try:
        PAGES[page](df, raw)
    except Exception as e:  # keep the app alive on unexpected data shapes
        st.error(f"Something went wrong in this module: {e}")
        with st.expander("Technical details"):
            st.exception(e)


main()
