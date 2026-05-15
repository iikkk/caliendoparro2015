"""
Trade Policy Counterfactual Explorer
=====================================
Streamlit app that visualises pre-computed general-equilibrium counterfactuals
for two tariff-change scenarios:

  • World  – tariff changes across all countries (1993 → 2005)
  • Bloc   – tariff changes only within the selected regional bloc

Data come from pre-computed MATLAB results (.mat) so the app is instant —
no heavy computation at runtime.

Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure the data/ package is importable when app.py is at the repo root
sys.path.insert(0, str(Path(__file__).parent / "data"))
from loader import build_country_df, build_sector_df, load_names  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trade Policy Counterfactual Explorer",
    page_icon="🌐",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .metric-card {
            background: #f8f9fb;
            border-radius: 10px;
            padding: 18px 22px;
            text-align: center;
        }
        .metric-label  { font-size: 0.82rem; color: #666; margin-bottom: 4px; }
        .metric-value  { font-size: 1.6rem; font-weight: 700; }
        .positive      { color: #1a9850; }
        .negative      { color: #d73027; }
        .neutral       { color: #555; }
        section[data-testid="stSidebar"] { background: #f0f4f8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

countries, tradeables, all_sectors = load_names()

with st.sidebar:
    st.title("🌐 Trade Explorer")
    st.caption("General-equilibrium counterfactuals — tariff changes 1993 → 2005")

    st.markdown("---")
    st.subheader("Country filter")
    selected_countries = st.multiselect(
        "Select countries to display",
        options=countries,
        default=countries,
        help="Hold Ctrl / Cmd to pick multiple countries.",
    )

    if not selected_countries:
        st.warning("Please select at least one country.")
        st.stop()

    selected_idx = [countries.index(c) for c in selected_countries]

    st.markdown("---")
    st.subheader("Display options")
    sort_by = st.selectbox(
        "Sort bars by",
        ["Country name", "World income change", "Bloc income change",
         "World export change", "Bloc export change"],
    )
    show_annotations = st.checkbox("Show value labels on bars", value=True)

    st.markdown("---")
    st.caption(
        "Data: Caliendo & Parro (2015) framework. "
        "Scenarios computed in MATLAB; loaded from pre-calculated `.mat` files."
    )

# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data
def get_country_df(idx_tuple):
    return build_country_df(list(idx_tuple))

df = get_country_df(tuple(selected_idx))

# Apply sort
sort_map = {
    "Country name":          "country",
    "World income change":   "income_world",
    "Bloc income change":    "income_bloc",
    "World export change":   "exports_world",
    "Bloc export change":    "exports_bloc",
}
df = df.sort_values(sort_map[sort_by], ascending=True).reset_index(drop=True)

# ── Helper: colour by sign ────────────────────────────────────────────────────

def sign_color(val):
    if val > 0.05:
        return "positive"
    if val < -0.05:
        return "negative"
    return "neutral"

def pct(v, decimals=2):
    return f"{v:+.{decimals}f}%"

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌐 Trade Policy Counterfactual Explorer")
st.markdown(
    f"Showing **{len(selected_countries)}** of {len(countries)} countries — "
    "comparing the **World** (all-country) and **Bloc-only** tariff-change scenarios."
)

# ── Summary metric cards ──────────────────────────────────────────────────────

st.subheader("Summary across selected countries")

avg_iw  = df["income_world"].mean()
avg_ib  = df["income_bloc"].mean()
avg_ew  = df["exports_world"].mean()
avg_eb  = df["exports_bloc"].mean()

cards = [
    ("Avg. Real Income Change\n(World tariffs)",  avg_iw,  "%"),
    ("Avg. Real Income Change\n(Bloc tariffs)",   avg_ib,  "%"),
    ("Avg. Export Change\n(World tariffs)",        avg_ew,  "%"),
    ("Avg. Export Change\n(Bloc tariffs)",         avg_eb,  "%"),
]

cols = st.columns(4)
for col, (label, val, unit) in zip(cols, cards):
    cls = sign_color(val)
    col.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label.replace(chr(10), "<br>")}</div>
          <div class="metric-value {cls}">{val:+.2f}{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Income & Wages", "📦 Trade Flows", "💰 Tariff Revenue", "🏭 Sector Analysis"]
)

# ─ Shared colour scale ───────────────────────────────────────────────────────
COLORS = {"World": "#2166ac", "Bloc": "#d6604d"}

def bar_chart(df_in, x_col, y_cols, names, title, x_label, colors=None):
    """Grouped horizontal bar chart with optional value labels."""
    fig = go.Figure()
    colors = colors or list(COLORS.values())
    for y_col, name, color in zip(y_cols, names, colors):
        fig.add_trace(go.Bar(
            y=df_in[x_col],
            x=df_in[y_col],
            name=name,
            orientation="h",
            marker_color=color,
            text=[f"{v:+.2f}%" for v in df_in[y_col]] if show_annotations else None,
            textposition="outside",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Country",
        barmode="group",
        height=max(350, len(df_in) * 32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee", zeroline=True, zerolinecolor="#aaa"),
    )
    # Red/green colouring for zero line
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
    return fig


# ── Tab 1: Income & Wages ────────────────────────────────────────────────────

with tab1:
    st.markdown("### Real income change (%) relative to base year")
    st.caption(
        "**World**: effects of 1993→2005 global tariff reductions. "
        "**Bloc**: effects of intra-bloc tariff reductions only."
    )

    fig_inc = bar_chart(
        df, "country",
        ["income_world", "income_bloc"],
        ["World tariffs", "Bloc tariffs"],
        "Real Income Change by Country",
        "% change",
    )
    st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Nominal wage change (%) — World scenario")

    fig_wg = go.Figure(go.Bar(
        y=df["country"],
        x=df["wages_world"],
        orientation="h",
        marker=dict(
            color=df["wages_world"],
            colorscale="RdYlGn",
            cmin=-5, cmax=5,
            colorbar=dict(title="% change"),
        ),
        text=[f"{v:+.2f}%" for v in df["wages_world"]] if show_annotations else None,
        textposition="outside",
    ))
    fig_wg.update_layout(
        title="Nominal Wage Change — World Tariff Scenario",
        xaxis_title="% change",
        yaxis_title="Country",
        height=max(350, len(df) * 32),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    fig_wg.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_wg, use_container_width=True)


# ── Tab 2: Trade Flows ───────────────────────────────────────────────────────

with tab2:
    st.markdown("### Export change (%) relative to base year")
    fig_ex = bar_chart(
        df, "country",
        ["exports_world", "exports_bloc"],
        ["World tariffs", "Bloc tariffs"],
        "Export Change by Country",
        "% change",
    )
    st.plotly_chart(fig_ex, use_container_width=True)

    if "imports_world" in df.columns and df["imports_world"].notna().any():
        st.markdown("---")
        st.markdown("### Import change (%) — World scenario")
        fig_im = go.Figure(go.Bar(
            y=df["country"],
            x=df["imports_world"],
            orientation="h",
            marker=dict(
                color=df["imports_world"],
                colorscale="Blues",
                colorbar=dict(title="% change"),
            ),
            text=[f"{v:+.2f}%" for v in df["imports_world"]] if show_annotations else None,
            textposition="outside",
        ))
        fig_im.update_layout(
            title="Import Change — World Tariff Scenario",
            xaxis_title="% change",
            yaxis_title="Country",
            height=max(350, len(df) * 32),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="#eee"),
        )
        fig_im.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_im, use_container_width=True)


# ── Tab 3: Tariff Revenue ────────────────────────────────────────────────────

with tab3:
    st.markdown("### Change in tariff revenue as % of value added")
    st.caption(
        "Negative values indicate a fall in tariff revenue (expected when tariffs drop)."
    )
    fig_tr = bar_chart(
        df, "country",
        ["tr_change_world", "tr_change_bloc"],
        ["World tariffs", "Bloc tariffs"],
        "Tariff Revenue Change (% of Value Added)",
        "percentage-point change",
        colors=["#4393c3", "#f4a582"],
    )
    st.plotly_chart(fig_tr, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Value added (counterfactual) — World scenario**")
        fig_va = px.bar(
            df, x="va_world", y="country",
            orientation="h",
            color="va_world",
            color_continuous_scale="Teal",
            labels={"va_world": "Value added (USD bn)", "country": ""},
            title="Counterfactual Value Added — World",
        )
        fig_va.update_layout(height=max(350, len(df) * 32),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_va, use_container_width=True)
    with col_b:
        st.markdown("**Value added (counterfactual) — Bloc scenario**")
        fig_vb = px.bar(
            df, x="va_bloc", y="country",
            orientation="h",
            color="va_bloc",
            color_continuous_scale="Teal",
            labels={"va_bloc": "Value added (USD bn)", "country": ""},
            title="Counterfactual Value Added — Bloc",
        )
        fig_vb.update_layout(height=max(350, len(df) * 32),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_vb, use_container_width=True)


# ── Tab 4: Sector Analysis ───────────────────────────────────────────────────

with tab4:
    st.markdown("### Sector-level export structure")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        sector_country = st.selectbox(
            "Select a country",
            options=selected_countries,
            key="sector_country",
        )
        sector_scenario = st.radio(
            "Scenario",
            ["World tariffs", "Bloc tariffs"],
            horizontal=True,
            key="sector_scenario",
        )

    cidx = countries.index(sector_country)
    scen = "world" if "World" in sector_scenario else "bloc"

    @st.cache_data
    def get_sector_df(cidx, scen):
        return build_sector_df(cidx, scen)

    sdf = get_sector_df(cidx, scen)

    if sdf.empty:
        st.info("No sector data available for this selection.")
    else:
        fig_sec = px.bar(
            sdf,
            x="exports",
            y="sector",
            color="tradeable",
            orientation="h",
            color_discrete_map={"Tradeable": "#2166ac", "Non-Tradeable": "#b2abd2"},
            labels={"exports": "Exports (USD bn)", "sector": ""},
            title=f"Sector Exports — {sector_country} ({sector_scenario})",
        )
        fig_sec.update_layout(
            height=max(450, len(sdf) * 24),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title="",
            xaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    st.markdown("---")
    st.markdown("### Cross-country sector heatmap")

    hm_scenario = st.radio(
        "Scenario for heatmap",
        ["World tariffs", "Bloc tariffs"],
        horizontal=True,
        key="hm_scenario",
    )
    hm_scen = "world" if "World" in hm_scenario else "bloc"

    @st.cache_data
    def get_heatmap_data(idx_tuple, scen):
        from loader import load_world, load_bloc
        _, tradeables, all_sectors = load_names()
        mat = load_world().get("Ejnp_all_out") if scen == "world" \
              else load_bloc().get("Ejnp_oN_out")
        if mat is None:
            return pd.DataFrame()
        idx = list(idx_tuple)
        cols = [countries[i] for i in idx]
        n_sec = len(all_sectors)
        sub = mat[:n_sec, :][:, idx]
        return pd.DataFrame(sub, index=all_sectors[:sub.shape[0]], columns=cols)

    hm_df = get_heatmap_data(tuple(selected_idx), hm_scen)

    if not hm_df.empty:
        fig_hm = px.imshow(
            hm_df,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels={"color": "Exports (USD bn)"},
            title=f"Sector × Country Export Heatmap — {hm_scenario}",
        )
        fig_hm.update_layout(
            height=max(450, len(hm_df) * 20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Country",
            yaxis_title="Sector",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────

with st.expander("📋 Raw data table"):
    display_df = df[[
        "country", "income_world", "income_bloc",
        "exports_world", "exports_bloc", "imports_world",
        "tr_change_world", "tr_change_bloc",
        "wages_world",
    ]].copy()
    display_df.columns = [
        "Country", "Income Δ World (%)", "Income Δ Bloc (%)",
        "Exports Δ World (%)", "Exports Δ Bloc (%)", "Imports Δ World (%)",
        "Tariff Rev. Δ World (pp)", "Tariff Rev. Δ Bloc (pp)",
        "Wages Δ World (%)",
    ]
    st.dataframe(
        display_df.style.format({c: "{:+.3f}" for c in display_df.columns[1:]}),
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download CSV",
        data=display_df.to_csv(index=False),
        file_name="trade_counterfactual_results.csv",
        mime="text/csv",
    )
