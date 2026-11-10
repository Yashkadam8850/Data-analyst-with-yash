"""Reporting: PDF report, executive summary, exports."""
import io
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .utils import detect_types, quality_report, df_to_csv_bytes, df_to_excel_bytes, need
from . import insights as ins_mod


def executive_summary(df, q, insights, ml=None):
    t = detect_types(df); lines = [f"The dataset contains {len(df):,} records and {df.shape[1]} columns ({len(t['numeric'])} numeric, "
                                    f"{len(t['categorical'])} categorical, {len(t['datetime'])} date). Data quality score: {q['score']}/100 "
                                    f"({q['missing_pct']:.2f}% missing cells, {q['duplicates']:,} duplicate rows)."]
    top = [i for i in insights if i["severity"] in ("high", "medium")][:3]
    if top: lines.append("Key concerns: " + "; ".join(i["title"] for i in top) + ".")
    opp = [i for i in insights if i["type"] in ("Trend", "Opportunity")][:3]
    if opp: lines.append("Key findings: " + "; ".join(i["title"] for i in opp) + ".")
    if ml: lines.append(f"Best predictive model for '{ml['target']}' is {ml['best']} ({ml['sort_col']} = {ml['results'].iloc[0][ml['sort_col']]:.3f}).")
    return "\n\n".join(lines)


def fig_png(fig):
    try: return fig.to_image(format="png", width=900, height=450, scale=2)
    except Exception: return None


def build_pdf(df, q, insights, ml, fig_titles, figs, title="IntelliAnalytics Report"):
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    ss = getSampleStyleSheet(); H = ParagraphStyle("H", parent=ss["Heading2"], textColor=colors.HexColor("#4f46e5")); body = ss["BodyText"]
    el = [Paragraph(title, ss["Title"]), Paragraph(datetime.now().strftime("Generated %Y-%m-%d %H:%M"), ss["Normal"]), Spacer(1, 12),
          Paragraph("Executive Summary", H)]
    for p in executive_summary(df, q, insights, ml).split("\n\n"): el += [Paragraph(p, body), Spacer(1, 6)]
    el.append(Paragraph("Data Profile", H))
    rows = [["Metric", "Value"], ["Records", f"{len(df):,}"], ["Columns", df.shape[1]], ["Missing cells", f"{q['missing']:,} ({q['missing_pct']:.2f}%)"],
            ["Duplicate rows", f"{q['duplicates']:,}"], ["Quality score", f"{q['score']}/100"]]
    def tbl(data, widths=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), .4, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")])]))
        return t
    el += [tbl(rows, [6 * cm, 9 * cm]), Spacer(1, 10)]
    num = detect_types(df)["numeric"][:8]
    if num:
        d = df[num].describe().T[["mean", "std", "min", "50%", "max"]].round(2); el.append(Paragraph("Numeric Summary", H))
        el += [tbl([["column"] + list(d.columns)] + [[i] + list(r) for i, r in d.iterrows()]), Spacer(1, 10)]
    el.append(Paragraph("Automated Insights & Recommendations", H))
    for i in insights[:15]:
        el.append(Paragraph(f"<b>[{i['type']}] {i['title']}</b><br/>{i['detail']}" + (f"<br/><i>Action: {i['action']}</i>" if i["action"] else ""), body)); el.append(Spacer(1, 4))
    if ml:
        el += [Paragraph(f"Model Performance – target '{ml['target']}'", H)]
        r = ml["results"].drop(columns=["Best_params"], errors="ignore").round(4)
        el.append(tbl([list(r.columns)] + r.values.tolist()))
    imgs = []
    for t in fig_titles:
        png = fig_png(figs[t])
        if png: imgs.append((t, png))
    if imgs:
        el.append(PageBreak()); el.append(Paragraph("Charts", H))
        for t, png in imgs: el += [Paragraph(t, body), Image(io.BytesIO(png), width=16 * cm, height=8 * cm), Spacer(1, 8)]
    doc.build(el); return buf.getvalue()


def render(df, df_raw):
    st.header("📄 Reporting & Export")
    q = quality_report(df); insights = ins_mod.generate(df, st.session_state.get("ml")); ml = st.session_state.get("ml")
    st.subheader("Executive summary"); st.info(executive_summary(df, q, insights, ml))
    figs = st.session_state.get("figs", {})
    sel = st.multiselect("Charts to include in the PDF / exports (visit other modules to generate more)", list(figs), default=list(figs)[:4])
    title = st.text_input("Report title", "IntelliAnalytics Report")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📑 Build PDF report", type="primary"):
            with st.spinner("Generating PDF..."):
                st.session_state["pdf"] = build_pdf(df, q, insights, ml, sel, figs, title)
        if st.session_state.get("pdf"): st.download_button("⬇️ Download PDF", st.session_state["pdf"], "intellianalytics_report.pdf", "application/pdf")
        st.caption("Charts appear in the PDF when the `kaleido` image engine is available.")
    with c2:
        st.download_button("⬇️ Cleaned data (CSV)", df_to_csv_bytes(df), "cleaned_data.csv", "text/csv")
        st.download_button("⬇️ Cleaned data (Excel)", df_to_excel_bytes(df), "cleaned_data.xlsx")
    with c3:
        if sel:
            z = io.BytesIO()
            with zipfile.ZipFile(z, "w") as zf:
                for t in sel:
                    safe = "".join(ch if ch.isalnum() else "_" for ch in t)
                    zf.writestr(f"{safe}.html", figs[t].to_html(include_plotlyjs="cdn"))
                    png = fig_png(figs[t]) if st.session_state.get("png_ok", True) else None
                    if png: zf.writestr(f"{safe}.png", png)
            st.download_button("⬇️ Charts (HTML + PNG zip)", z.getvalue(), "charts.zip", "application/zip")
        else: st.caption("No charts generated yet.")
    st.subheader("Export insights")
    st.download_button("⬇️ Insights (CSV)", pd.DataFrame(insights).to_csv(index=False).encode(), "insights.csv")
