"""Shared helpers: loading, type detection, quality scoring, UI primitives."""
import io
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

ROLE_KEYWORDS = {
    "revenue": ["revenue", "sales", "amount", "turnover", "income", "total"],
    "cost": ["cost", "expense", "cogs"],
    "profit": ["profit", "margin", "earning"],
    "customer": ["customer", "client", "user", "cust", "member", "account"],
    "order": ["order", "invoice", "transaction", "txn"],
    "quantity": ["quantity", "qty", "units", "volume"],
}


# ------------------------------------------------------------------ loading
def excel_sheets(file) -> list:
    return pd.ExcelFile(file).sheet_names


def load_file(file, sheet=None) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(file, sheet_name=sheet or 0)
    else:
        raw = file.getvalue() if hasattr(file, "getvalue") else open(file, "rb").read()
        df, err = None, None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                break
            except Exception as e:  # noqa
                err = e
        if df is None:
            raise ValueError(f"Could not parse file: {err}")
    return auto_coerce(df)


def auto_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy names and convert text columns that are really numbers/dates."""
    df = df.copy()
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    seen = {}
    for i, c in enumerate(df.columns):  # de-duplicate column names
        seen[c] = seen.get(c, -1) + 1
        if seen[c]:
            df.columns.values[i] = f"{c}_{seen[c]}"
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("string", "str"):
            s = s.astype(object).where(s.notna(), None)
            nn = s.dropna().astype(str).str.strip()
            if nn.empty:
                continue
            cleaned = nn.str.replace(r"[,$€£%\s]", "", regex=True)
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().mean() > 0.95 and nn.str.contains(r"\d").mean() > 0.95:
                df[c] = pd.to_numeric(s.astype(str).str.replace(r"[,$€£%\s]", "", regex=True), errors="coerce")
                continue
            sample = nn.sample(min(200, len(nn)), random_state=0)
            if sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}").mean() > 0.9:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[c] = pd.to_datetime(s, errors="coerce")
    return df


# ------------------------------------------------------------- type detection
def detect_types(df: pd.DataFrame) -> dict:
    num, cat, dt, idc = [], [], [], []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt.append(c)
        elif pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            nun = s.nunique(dropna=True)
            is_int = pd.api.types.is_integer_dtype(s) or (s.dropna() % 1 == 0).all()
            looks_id = is_int and re.search(r"(^|[_\s])(id|key|code|no|number)$", c.lower()) is not None
            (idc if looks_id else num).append(c)
        else:
            nun = s.nunique(dropna=True)
            (idc if nun / n > 0.9 and nun > 20 else cat).append(c)
    return {"numeric": num, "categorical": cat, "datetime": dt, "id": idc}


def cat_like(df, types, max_levels=15):
    """Categorical columns plus low-cardinality numerics (usable as groups)."""
    low = [c for c in types["numeric"] if df[c].nunique() <= max_levels]
    return [c for c in types["categorical"] if df[c].nunique() <= 60] + low


def guess_col(cols, role, exclude=()):
    for kw in ROLE_KEYWORDS.get(role, []):
        for c in cols:
            if c not in exclude and kw in c.lower():
                return c
    return None


def idx_of(options, value, default=0):
    return options.index(value) if value in options else default


# ------------------------------------------------------------- quality
def outlier_mask(s: pd.Series, method="IQR", thresh=None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if method == "Z-score":
        z = (s - s.mean()) / (s.std(ddof=0) or 1)
        return z.abs() > (thresh or 3)
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    k = thresh or 1.5
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def quality_report(df: pd.DataFrame, types=None) -> dict:
    types = types or detect_types(df)
    cells = max(df.shape[0] * df.shape[1], 1)
    miss = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())
    completeness = 1 - miss / cells
    uniqueness = 1 - dup / max(len(df), 1)
    constant = sum(df[c].nunique(dropna=True) <= 1 for c in df.columns)
    consistency = 1 - constant / max(df.shape[1], 1)
    out_share = []
    for c in types["numeric"]:
        if df[c].nunique() > 5:
            out_share.append(outlier_mask(df[c]).mean())
    validity = 1 - (np.mean(out_share) if out_share else 0)
    score = 100 * (.4 * completeness + .2 * uniqueness + .2 * consistency + .2 * validity)
    return dict(score=round(score, 1), missing=miss, missing_pct=100 * miss / cells, duplicates=dup,
                completeness=completeness, uniqueness=uniqueness, consistency=consistency, validity=validity,
                constant_cols=constant)


def memory_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


# ----------------------------------------------------------------- UI
def theme_template():
    return "plotly_dark" if st.session_state.get("theme", "Light") == "Dark" else "plotly_white"


def apply_theme_css():
    dark = st.session_state.get("theme", "Light") == "Dark"
    bg, fg, card, border = ("#0e1117", "#e5e7eb", "#1a1f2b", "#2d3446") if dark else ("#f6f7fb", "#111827", "#ffffff", "#e5e7eb")
    st.markdown(f"""
    <style>
    .stApp {{ background:{bg}; color:{fg}; }}
    section[data-testid="stSidebar"] {{ background:{card}; border-right:1px solid {border}; }}
    .stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li {{ color:{fg}; }}
    .kpi {{ background:{card}; border:1px solid {border}; border-left:5px solid #4f46e5; border-radius:12px;
            padding:14px 16px; margin-bottom:10px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
    .kpi .l {{ font-size:.78rem; opacity:.7; text-transform:uppercase; letter-spacing:.04em; }}
    .kpi .v {{ font-size:1.55rem; font-weight:700; line-height:1.25; }}
    .kpi .d {{ font-size:.8rem; opacity:.8; }}
    .insight {{ background:{card}; border:1px solid {border}; border-radius:10px; padding:10px 14px; margin:6px 0; }}
    .block-container {{ padding-top:1.5rem; max-width:1400px; }}
    </style>""", unsafe_allow_html=True)


def kpi(label, value, delta="", color="#4f46e5"):
    st.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="l">{label}</div>'
                f'<div class="v">{value}</div><div class="d">{delta}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            kpi(*it)


def show(fig, title, key=None, height=420):
    """Render a Plotly figure with theme, register it for export."""
    fig.update_layout(template=theme_template(), title=title, height=height,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.session_state.setdefault("figs", {})[title] = fig
    st.plotly_chart(fig, key=key or f"pc_{abs(hash(title))}")


def human(x, prefix=""):
    try:
        x = float(x)
    except Exception:
        return str(x)
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"{prefix}{x / d:,.2f}{unit}"
    return f"{prefix}{x:,.2f}"


def need(cond, msg):
    """Stop rendering a section gracefully if a precondition fails."""
    if not cond:
        st.info(msg)
        st.stop()


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Cleaned")
    return buf.getvalue()
