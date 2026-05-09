import streamlit as st
import base64
import json
import pandas as pd
from pathlib import Path
from utils.backend import MechAnalyzerBackend
from utils.ml_engine import MLEngine
from utils.insights import InsightsGenerator

st.set_page_config(
    page_title="MechVision AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.main { background: #F7F6F2; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

/* Hero */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem; color: #1A1A1A; line-height: 1.15;
    margin: 0; letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.05rem; color: #6B6B6B; font-weight: 300;
    margin-top: 0.5rem; max-width: 560px; line-height: 1.7;
}
.badge-pill {
    display: inline-block; background: #E8F4EC;
    color: #1E6B3C; font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 1rem;
}

/* Cards */
.metric-card {
    background: #FFFFFF; border-radius: 14px;
    padding: 1.4rem 1.6rem; border: 1px solid #EBEBEB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-card .label { font-size: 11px; color: #9A9A9A; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.metric-card .value { font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; color: #1A1A1A; line-height: 1; }
.metric-card .delta { font-size: 12px; color: #1E6B3C; font-weight: 500; margin-top: 4px; }

/* Section headings */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem; color: #1A1A1A; margin: 2rem 0 0.3rem;
}
.section-sub { font-size: 0.9rem; color: #888; margin-bottom: 1.2rem; }

/* Tables */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th {
    background: #F0EFE9; color: #555; font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
    padding: 10px 14px; text-align: left; border: none;
}
.styled-table td {
    padding: 10px 14px; border-bottom: 1px solid #F0EFE9;
    color: #2A2A2A; vertical-align: middle;
}
.styled-table tr:hover td { background: #FAFAF7; }
.styled-table tr:last-child td { border-bottom: none; }

/* Accuracy ring placeholder */
.acc-ring {
    width: 100px; height: 100px; border-radius: 50%;
    border: 8px solid #E8F4EC; border-top-color: #1E6B3C;
    display: inline-block;
}

/* Tag pills */
.tag { display: inline-block; border-radius: 8px; padding: 2px 10px;
    font-size: 11px; font-weight: 600; margin: 2px; }
.tag-green  { background:#E8F4EC; color:#1E6B3C; }
.tag-blue   { background:#E8F0FB; color:#1A4DB0; }
.tag-orange { background:#FEF3E8; color:#B05A1A; }
.tag-red    { background:#FDECEC; color:#B01A1A; }
.tag-purple { background:#F0ECFD; color:#5B1AB0; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
    border-radius: 16px; padding: 1.5rem 1.8rem; color: #F7F6F2;
    margin-bottom: 1rem;
}
.insight-box .i-title { font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; margin-bottom: 0.5rem; color: #E8F4EC; }
.insight-box .i-body { font-size: 0.88rem; line-height: 1.7; color: #C8C8C8; }

/* Model selector */
.model-btn { cursor: pointer; border: 2px solid #EBEBEB;
    border-radius: 12px; padding: 1rem 1.2rem; background: white;
    transition: all 0.2s; text-align: center; }
.model-btn:hover { border-color: #1A1A1A; }
.model-btn.selected { border-color: #1E6B3C; background: #E8F4EC; }

/* Stapp button overrides */
.stButton > button {
    background: #1A1A1A; color: #F7F6F2; border: none;
    border-radius: 10px; padding: 0.65rem 1.6rem;
    font-family: 'Outfit', sans-serif; font-weight: 500;
    font-size: 14px; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; background: #1A1A1A; color: #F7F6F2; }

.stSelectbox label, .stSlider label { font-size: 13px !important; color: #555 !important; }

hr { border: none; border-top: 1px solid #EBEBEB; margin: 1.5rem 0; }

.mono { font-family: 'DM Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in {
    "extracted_data": None,
    "df": None,
    "ml_results": None,
    "insights": None,
    "step": 1,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

backend = MechAnalyzerBackend()
ml_engine = MLEngine()
ig = InsightsGenerator()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <span style='font-family:"DM Serif Display",serif;font-size:1.4rem;color:#1A1A1A;'>⚙️ MechVision AI</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Pipeline steps**")

    steps = ["📤 Upload diagram", "🔍 Extract data", "🤖 Run ML model", "📊 Business insights"]
    for i, s in enumerate(steps, 1):
        color = "#1E6B3C" if st.session_state.step >= i else "#BBBBBB"
        bg    = "#E8F4EC"  if st.session_state.step >= i else "#F5F5F5"
        st.markdown(
            f"<div style='padding:8px 12px;border-radius:8px;background:{bg};"
            f"color:{color};font-size:13px;font-weight:500;margin-bottom:4px;'>{s}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**ML Model**")
    model_choice = st.selectbox(
        "Choose algorithm",
        ["Linear Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"],
        label_visibility="collapsed",
    )
    st.markdown("**Target variable**")
    target_col = st.selectbox(
        "Target",
        ["complexity_score", "est_size_cm", "connection_count"],
        label_visibility="collapsed",
    )
    st.markdown("**Test size**")
    test_size = st.slider("", 0.1, 0.4, 0.2, 0.05, label_visibility="collapsed")

    st.markdown("---")
    if st.button("🔄  Reset session"):
        for k in ["extracted_data", "df", "ml_results", "insights", "step"]:
            st.session_state[k] = None if k != "step" else 1
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<span class="badge-pill">AI-Powered Engineering Intelligence</span>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Mechanical Diagram<br>Intelligence System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload any engineering schematic. Claude AI extracts structured data, '
    'runs machine learning models, and translates results into plain-language business insights.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-head">Step 1 — Upload diagram</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Supports engineering schematics, patent drawings, P&ID diagrams, CAD exports (PNG/JPG/WEBP)</p>', unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

if uploaded:
    col_img, col_info = st.columns([1, 1], gap="large")
    with col_img:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
    with col_info:
        fsize = uploaded.size / 1024
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
            <div class="label">File</div>
            <div style="font-size:1rem;font-weight:600;color:#1A1A1A;">{uploaded.name}</div>
            <div class="delta">{fsize:.1f} KB · {uploaded.type}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F7F6F2;border-radius:12px;padding:1rem 1.2rem;border:1px solid #EBEBEB;font-size:13px;color:#555;line-height:1.7;">
            ✅ Image loaded successfully<br>
            🔍 Claude will identify all labeled components<br>
            📐 Subsystem classification is automatic<br>
            🔢 Numeric features extracted per component
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — EXTRACT
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-head">Step 2 — Extract structured data</p>', unsafe_allow_html=True)

    if st.button("🔍  Analyze diagram with Claude AI"):
        st.session_state.step = 2
        img_bytes = uploaded.read()
        b64 = base64.b64encode(img_bytes).decode()
        mime = uploaded.type

        with st.spinner("Claude is reading the diagram…"):
            result = backend.extract_from_image(b64, mime)

        if result:
            st.session_state.extracted_data = result
            st.session_state.df = pd.DataFrame(result["components"])
            st.success(f"✅ Extracted {len(result['components'])} components from the diagram")
        else:
            st.error("Extraction failed. Check your API key in secrets.")

    if st.session_state.df is not None:
        df = st.session_state.df
        result = st.session_state.extracted_data

        # — KPI row ——————————————————————————————————————————————
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Components found</div>
                <div class="value">{len(df)}</div>
                <div class="delta">from diagram labels</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            subs = df["subsystem"].nunique() if "subsystem" in df.columns else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Subsystems</div>
                <div class="value">{subs}</div>
                <div class="delta">distinct categories</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            avg_cx = f"{df['complexity_score'].mean():.1f}" if "complexity_score" in df.columns else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Avg complexity</div>
                <div class="value">{avg_cx}</div>
                <div class="delta">out of 5.0</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            actuated = int(df["is_actuated"].sum()) if "is_actuated" in df.columns else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Actuated parts</div>
                <div class="value">{actuated}</div>
                <div class="delta">require power input</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # — Component table ——————————————————————————————————————
        st.markdown("**Extracted component table**")
        st.markdown('<p class="section-sub">Every labeled component with AI-assigned features</p>', unsafe_allow_html=True)

        tag_colors = {
            "Control": "tag-blue", "Mechanical": "tag-green",
            "Fluid": "tag-purple", "Safety": "tag-red", "Power": "tag-orange",
        }

        rows_html = ""
        for _, row in df.iterrows():
            sub = row.get("subsystem", "—")
            tag_cls = tag_colors.get(sub, "tag-green")
            cx = row.get("complexity_score", 0)
            cx_bar = "●" * int(round(cx)) + "○" * (5 - int(round(cx)))
            act = "✅" if row.get("is_actuated", 0) else "—"
            rows_html += f"""
            <tr>
                <td class='mono'>{row.get('ref_number','')}</td>
                <td><b>{row.get('name','')}</b></td>
                <td><span class='tag {tag_cls}'>{sub}</span></td>
                <td style='text-align:center'>{row.get('connection_count','')}</td>
                <td class='mono'>{cx_bar} {cx:.1f}</td>
                <td style='text-align:center'>{row.get('est_size_cm','—')} cm</td>
                <td style='text-align:center'>{act}</td>
            </tr>"""

        st.markdown(f"""
        <div style='overflow-x:auto;border-radius:12px;border:1px solid #EBEBEB;'>
        <table class='styled-table'>
            <thead><tr>
                <th>Ref #</th><th>Component name</th><th>Subsystem</th>
                <th>Connections</th><th>Complexity</th><th>Est. size</th><th>Actuated</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # — Subsystem breakdown table ————————————————————————————
        st.markdown("**Subsystem summary**")
        if "subsystem" in df.columns:
            grp = df.groupby("subsystem").agg(
                Components=("name", "count"),
                Avg_Complexity=("complexity_score", "mean"),
                Total_Connections=("connection_count", "sum"),
                Actuated=("is_actuated", "sum"),
            ).reset_index().rename(columns={"subsystem": "Subsystem"})

            grp["Avg_Complexity"] = grp["Avg_Complexity"].round(2)
            grp["Risk_Flag"] = grp["Avg_Complexity"].apply(
                lambda x: "🔴 High" if x >= 4 else ("🟡 Medium" if x >= 2.5 else "🟢 Low")
            )
            st.dataframe(grp, use_container_width=True, hide_index=True)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button("⬇️  Download extracted data (CSV)", csv,
                           "mech_components.csv", "text/csv")

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — ML
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<p class="section-head">Step 3 — Machine learning model</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-sub">Running <b>{model_choice}</b> · predicting <b>{target_col}</b> · test split {int(test_size*100)}%</p>', unsafe_allow_html=True)

        if st.button(f"🤖  Train {model_choice}"):
            st.session_state.step = 3
            with st.spinner("Training model…"):
                res = ml_engine.run(df, model_choice, target_col, test_size)
            st.session_state.ml_results = res
            if "error" in res:
                st.error(res["error"])

        if st.session_state.ml_results:
            res = st.session_state.ml_results
            if "error" not in res:

                # — Accuracy KPIs ————————————————————————————————
                st.markdown("**Model performance**")
                a1, a2, a3, a4 = st.columns(4)
                metrics = res.get("metrics", {})

                def mcard(col, label, val, delta="", good=True):
                    clr = "#1E6B3C" if good else "#B01A1A"
                    col.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value" style="color:{clr};font-size:1.9rem;">{val}</div>
                        <div class="delta">{delta}</div>
                    </div>""", unsafe_allow_html=True)

                r2 = metrics.get("R2", metrics.get("Accuracy", 0))
                r2_good = r2 >= 0.5
                mcard(a1, "R² / Accuracy", f"{r2:.3f}", "explained variance" if "R2" in metrics else "correct predictions", r2_good)
                if "RMSE" in metrics:
                    mcard(a2, "RMSE", f"{metrics['RMSE']:.3f}", "root mean sq error", metrics['RMSE'] < 1.0)
                if "MAE" in metrics:
                    mcard(a3, "MAE", f"{metrics['MAE']:.3f}", "mean abs error", metrics['MAE'] < 1.0)
                samples = res.get("samples", {})
                mcard(a4, "Train / Test", f"{samples.get('train','-')} / {samples.get('test','-')}", "sample split", True)

                st.markdown("<br>", unsafe_allow_html=True)

                # — Predictions table ————————————————————————————
                st.markdown("**Predicted vs actual values**")
                pred_df = res.get("predictions_df")
                if pred_df is not None:
                    st.dataframe(pred_df.style.format({
                        "Actual": "{:.2f}", "Predicted": "{:.2f}", "Error": "{:.2f}", "Error%": "{:.1f}%"
                    }).background_gradient(subset=["Error%"], cmap="RdYlGn_r"),
                    use_container_width=True, hide_index=True)

                # — Feature importance ————————————————————————————
                fi = res.get("feature_importance")
                if fi:
                    st.markdown("**Feature importance**")
                    fi_df = pd.DataFrame(fi).sort_values("Importance", ascending=False)
                    st.dataframe(fi_df.style.background_gradient(subset=["Importance"], cmap="Greens"),
                                 use_container_width=True, hide_index=True)

                # — Coefficients (LR only) ————————————————————————
                coefs = res.get("coefficients")
                if coefs:
                    st.markdown("**Model coefficients**")
                    coef_df = pd.DataFrame(coefs)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # ══════════════════════════════════════════════════════════════
                # STEP 4 — INSIGHTS
                # ══════════════════════════════════════════════════════════════
                st.markdown('<p class="section-head">Step 4 — Business insights</p>', unsafe_allow_html=True)
                st.markdown('<p class="section-sub">Plain-language summary for non-technical stakeholders</p>', unsafe_allow_html=True)

                if st.button("💡  Generate business insights"):
                    st.session_state.step = 4
                    with st.spinner("Generating insights…"):
                        insights = ig.generate(df, res, model_choice, target_col)
                    st.session_state.insights = insights

                if st.session_state.insights:
                    ins = st.session_state.insights
                    for item in ins.get("insights", []):
                        emoji = {"risk": "🔴", "opportunity": "🟢", "observation": "🔵"}.get(item.get("type",""), "⚪")
                        st.markdown(f"""
                        <div class="insight-box">
                            <div class="i-title">{emoji} {item.get('title','')}</div>
                            <div class="i-body">{item.get('body','')}</div>
                        </div>""", unsafe_allow_html=True)

                    # — Executive summary table ——————————————————
                    st.markdown("**Executive summary table**")
                    exec_rows = ins.get("exec_table", [])
                    if exec_rows:
                        exec_df = pd.DataFrame(exec_rows)
                        st.dataframe(exec_df, use_container_width=True, hide_index=True)

                    # — Recommended actions ——————————————————————
                    actions = ins.get("actions", [])
                    if actions:
                        st.markdown("**Recommended actions**")
                        for i, a in enumerate(actions, 1):
                            priority_color = {"High": "#B01A1A", "Medium": "#B05A1A", "Low": "#1E6B3C"}.get(a.get("priority",""), "#555")
                            st.markdown(f"""
                            <div style='display:flex;align-items:flex-start;gap:12px;padding:12px 0;
                                border-bottom:1px solid #F0EFE9;'>
                                <div style='background:#F0EFE9;border-radius:8px;padding:4px 10px;
                                    font-family:"DM Mono",monospace;font-size:12px;
                                    font-weight:600;color:#1E6B3C;flex-shrink:0;'>{i:02d}</div>
                                <div style='flex:1;'>
                                    <div style='font-weight:600;font-size:14px;color:#1A1A1A;'>
                                        {a.get('action','')}
                                    </div>
                                    <div style='font-size:12px;color:#888;margin-top:3px;'>
                                        {a.get('reason','')}
                                    </div>
                                </div>
                                <span style='font-size:11px;font-weight:600;color:{priority_color};
                                    background:{priority_color}18;padding:3px 10px;border-radius:20px;flex-shrink:0;'>
                                    {a.get('priority','')} priority
                                </span>
                            </div>""", unsafe_allow_html=True)

                    # Download insights report
                    report = ig.to_report(df, res, ins)
                    st.download_button("⬇️  Download full report (TXT)", report,
                                       "mech_insights_report.txt", "text/plain")

else:
    st.markdown("""
    <div style='background:#FFFFFF;border:2px dashed #DEDED8;border-radius:16px;
        padding:3rem;text-align:center;color:#AAAAAA;'>
        <div style='font-size:3rem;margin-bottom:0.5rem;'>📐</div>
        <div style='font-size:1rem;font-weight:500;color:#666;'>Upload an engineering diagram above to begin</div>
        <div style='font-size:13px;margin-top:0.5rem;'>Supports patent drawings, P&ID schematics, CAD exports</div>
    </div>
    """, unsafe_allow_html=True)
