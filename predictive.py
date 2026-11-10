"""Predictive analytics: time-series / sales / demand forecasting and churn / risk prediction."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBRegressor

from .utils import detect_types, show, need, guess_col, idx_of, human
from . import ml

FREQ = {"Daily": ("D", 7), "Weekly": ("W", 52), "Monthly": ("MS", 12), "Quarterly": ("QS", 4)}


def prep_series(df, date, value, freq, agg):
    s = df[[date, value]].dropna().set_index(date)[value].sort_index().resample(freq).agg(agg)
    if agg == "sum":
        s = s.where(s.index.isin(s.index), 0)
    return s.interpolate(limit_direction="both").astype(float)


def _lag_frame(s, k):
    d = pd.DataFrame({f"lag{i}": s.shift(i) for i in range(1, k + 1)}); d["y"] = s
    return d.dropna()


def _ml_forecast(train, h, k):
    d = _lag_frame(train, k); m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=42)
    m.fit(d.drop(columns="y"), d.y); hist = list(train.values); out = []
    for _ in range(h):
        x = pd.DataFrame([[hist[-i] for i in range(1, k + 1)]], columns=[f"lag{i}" for i in range(1, k + 1)])
        p = float(m.predict(x)[0]); out.append(p); hist.append(p)
    resid = d.y - m.predict(d.drop(columns="y"))
    return np.array(out), float(np.std(resid))


def _fit_predict(name, train, h, sp):
    n = len(train)
    if name == "Seasonal naive":
        p = sp if sp and n >= 2 * sp else 1
        return np.array([train.iloc[-p + (i % p)] for i in range(h)]), float(train.diff().std() or 1)
    if name == "Holt-Winters (ETS)":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seas = "add" if sp and n >= 2 * sp else None
        m = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=seas, seasonal_periods=sp if seas else None).fit()
        return np.asarray(m.forecast(h)), float(np.std(m.resid))
    if name == "SARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seas = (0, 1, 1, sp) if sp and sp <= 12 and n >= 3 * sp else (0, 0, 0, 0)
        m = SARIMAX(train, order=(1, 1, 1), seasonal_order=seas, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return np.asarray(m.forecast(h)), float(np.std(m.resid[1:]))
    if name == "XGBoost (lag features)":
        return _ml_forecast(train, h, max(2, min(12, n // 4)))
    raise ValueError(name)


def _mape(a, f):
    a, f = np.asarray(a), np.asarray(f); mask = a != 0
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100) if mask.any() else np.nan


def forecast_series(s, horizon, sp, models=None):
    models = models or ["Seasonal naive", "Holt-Winters (ETS)", "SARIMA", "XGBoost (lag features)"]
    h_test = max(2, min(horizon, len(s) // 4)); train, test = s.iloc[:-h_test], s.iloc[-h_test:]
    rows = []
    for m in models:
        try:
            f, _ = _fit_predict(m, train, h_test, sp)
            rows.append(dict(Model=m, MAPE_pct=_mape(test, f), RMSE=float(np.sqrt(np.mean((test.values - f) ** 2))), MAE=float(np.mean(np.abs(test.values - f)))))
        except Exception as e:
            rows.append(dict(Model=m, MAPE_pct=np.nan, RMSE=np.nan, MAE=np.nan, error=str(e)[:80]))
    bt = pd.DataFrame(rows).sort_values("RMSE"); best = bt.dropna(subset=["RMSE"]).Model.iloc[0]
    f, sd = _fit_predict(best, s, horizon, sp)
    idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=s.index.freq or pd.infer_freq(s.index))[1:]
    width = 1.96 * sd * np.sqrt(np.arange(1, horizon + 1))
    return bt, best, pd.DataFrame({"forecast": f, "lower": f - width, "upper": f + width}, index=idx)


def render(df):
    st.header("🔮 Predictive Analytics")
    types = detect_types(df)
    tabs = st.tabs(["Time-series / Sales / Demand forecasting", "Churn & Risk prediction"])

    with tabs[0]:
        need(types["datetime"] and types["numeric"], "Forecasting needs a date column and a numeric column.")
        use = st.radio("Use case", ["Sales forecasting", "Demand forecasting", "General time series"], horizontal=True)
        role = {"Sales forecasting": "revenue", "Demand forecasting": "quantity"}.get(use, "revenue")
        c1, c2, c3, c4 = st.columns(4)
        dc = c1.selectbox("Date", types["datetime"], key="f_d")
        g = guess_col(types["numeric"], role); v = c2.selectbox("Value", types["numeric"], index=idx_of(types["numeric"], g), key="f_v")
        fr = c3.selectbox("Frequency", list(FREQ), index=2, key="f_f"); ag = c4.selectbox("Aggregation", ["sum", "mean"], key="f_a")
        horizon = st.slider("Forecast horizon (periods)", 3, 36, 12)
        freq, sp = FREQ[fr]; s = prep_series(df, dc, v, freq, ag)
        need(len(s) >= 12, f"Only {len(s)} periods after aggregation; need at least 12. Try a finer frequency.")
        if st.button("Run forecast", type="primary"):
            with st.spinner("Backtesting models..."):
                bt, best, fc = forecast_series(s, horizon, sp)
            st.session_state["forecast"] = dict(bt=bt, best=best, fc=fc, s=s, v=v, use=use)
        r = st.session_state.get("forecast")
        if r and r["v"] == v:
            st.dataframe(r["bt"].round(3)); st.success(f"Best model by backtest RMSE: **{r['best']}**")
            fc, s_ = r["fc"], r["s"]; fig = go.Figure(go.Scatter(x=s_.index, y=s_.values, name="history"))
            fig.add_scatter(x=fc.index, y=fc.forecast, name="forecast", line=dict(dash="dash"))
            fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.upper) + list(fc.lower[::-1]), fill="toself", name="95% interval", line=dict(width=0), opacity=.25)
            show(fig, f"{r['use']} – {v}")
            hist_avg = s_.iloc[-len(fc):].mean()
            st.write(f"Forecast total next {len(fc)} periods: **{human(fc.forecast.sum())}** · average vs last {len(fc)} periods: **{100 * (fc.forecast.mean() / hist_avg - 1):+.1f}%**")
            st.dataframe(fc.round(2)); st.download_button("Download forecast", fc.to_csv().encode(), "forecast.csv")

    with tabs[1]:
        use = st.radio("Use case", ["Customer churn", "Risk prediction"], horizontal=True)
        cands = [c for c in df.columns if c not in types["id"] and 2 <= df[c].nunique() <= 2]
        need(cands, "No binary (2-class) column found to use as the outcome (e.g. churned yes/no, default 0/1).")
        gt = guess_col(cands, "x") or next((c for c in cands if any(k in c.lower() for k in ["churn", "default", "risk", "fraud", "attrit", "cancel"])), cands[0])
        target = st.selectbox("Outcome column", cands, index=idx_of(cands, gt))
        classes = sorted(df[target].dropna().astype(str).unique()); pos = st.selectbox("Positive class (the event to predict)", classes, index=len(classes) - 1)
        excl = st.multiselect("Exclude features (leakage guard)", [c for c in df.columns if c != target], key="cr_ex")
        if st.button("Train predictive model", type="primary"):
            with st.spinner("Training with leakage-safe pipeline..."):
                st.session_state["risk"] = ml.train_and_compare(df, target, excl, None, .2, 5, "Random", balanced=True)
                st.session_state["risk_pos"] = pos
        r = st.session_state.get("risk")
        if r and r["target"] == target:
            res = r["results"]; st.dataframe(res.drop(columns="Best_params").round(4))
            best = r["best"]; pipe = r["pipes"][best]; classes_ = list(r["label_encoder"].classes_); pi = classes_.index(st.session_state.get("risk_pos", pos))
            st.success(f"Best model: **{best}** · F1 {res.F1.iloc[0]:.3f} · ROC-AUC {res.ROC_AUC.iloc[0]:.3f}")
            # Out-of-fold scores for ALL rows (no row is scored by a model that saw it)
            d = df.dropna(subset=[target]); X, _, _, _ = ml.prepare_features(d, target, excl)
            y = pd.Series(r["label_encoder"].transform(d[target].astype(str)), index=d.index)
            try:
                oof = cross_val_predict(clone(pipe), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, pi]
            except Exception as e:
                st.error(f"Scoring failed: {e}"); st.stop()
            sc = d.copy(); sc["risk_score"] = oof
            sc["risk_tier"] = pd.cut(sc.risk_score, [-.01, .33, .66, 1.0], labels=["Low", "Medium", "High"])
            a, b = st.columns(2)
            with a: show(px.histogram(sc, x="risk_score", nbins=30, color="risk_tier"), "Risk score distribution", height=350)
            with b:
                t = sc.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).reset_index(); t.columns = ["tier", "count"]
                show(px.bar(t, x="tier", y="count", color="tier"), "Records per risk tier", height=350)
            st.write(f"**{(sc.risk_tier == 'High').sum():,}** high-risk records ({100 * (sc.risk_tier == 'High').mean():.1f}%). Top 20 highest risk:")
            st.dataframe(sc.sort_values("risk_score", ascending=False).head(20))
            st.download_button("Download scored data", sc.to_csv(index=False).encode(), "risk_scores.csv")
            st.session_state["scored"] = sc
