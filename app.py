import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configure page layout
st.set_page_config(
    page_title="CCTS & EU-CBAM Carbon-VaR Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Indian Carbon Market (CCTS) & EU-CBAM Exposure Model")
st.caption("Quantitative Corporate Earnings-at-Risk Engine | Compliance Mechanics under BEE & Regulation (EU) 2023/956")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🕹️ Market & Policy Drivers")

p_ccc = st.sidebar.slider(
    "Domestic CCC Price (₹/tCO2e)",
    min_value=500,
    max_value=3000,
    value=1200,
    step=100
)

p_eu_ets = st.sidebar.slider(
    "EU-ETS Allowance Price (€/tCO2e)",
    min_value=60.0,
    max_value=120.0,
    value=85.0,
    step=5.0
)

eur_inr = st.sidebar.number_input(
    "FX Rate (EUR/INR)",
    min_value=80.0,
    max_value=110.0,
    value=92.0,
    step=0.5
)

delta_art9 = st.sidebar.toggle(
    "Recognize Article 9 Domestic Deduction (δ_Art9 = 1)",
    value=False,
    help="When enabled, domestic carbon costs paid under CCTS are deducted from the EU-CBAM duty."
)

indirect_expansion = st.sidebar.toggle(
    "Stress Annex II: Include Indirect Power in CBAM",
    value=False,
    help="Models downside tail-risk if Brussels expands CBAM to tax captive coal electricity for aluminum."
)

# ----------------- EMPIRICAL DATASET -----------------
# Grounded in SEBI BRSR Principle 6 Disclosures
cohort_data = [
    {
        "ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd.", "sector": "Steel (BF-BOF)",
        "output_mt": 23.430, "scope1_mt": 54.100, "scope2_mt": 2.200,
        "actual_gei": 2.403, "cbam_direct_gei": 2.309, "target_gei": 2.313,
        "ebitda_cr": 34848.0, "eu_export_mt": 1.350, "bm_eu": 1.370
    },
    {
        "ticker": "JSWSTEEL.NS", "name": "JSW Steel Ltd.", "sector": "Steel (BF-BOF/DRI)",
        "output_mt": 26.850, "scope1_mt": 60.259, "scope2_mt": 1.023,
        "actual_gei": 2.282, "cbam_direct_gei": 2.244, "target_gei": 2.230,
        "ebitda_cr": 29821.0, "eu_export_mt": 2.150, "bm_eu": 1.370
    },
    {
        "ticker": "JINDALSTEL.NS", "name": "Jindal Steel Ltd.", "sector": "Steel (Coal-DRI/BF)",
        "output_mt": 12.850, "scope1_mt": 35.630, "scope2_mt": 0.690,
        "actual_gei": 2.826, "cbam_direct_gei": 2.773, "target_gei": 2.623,
        "ebitda_cr": 9099.0, "eu_export_mt": 0.420, "bm_eu": 1.370
    },
    {
        "ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Cement (Clinker/Grey)",
        "output_mt": 145.000, "scope1_mt": 88.884, "scope2_mt": 2.040,
        "actual_gei": 0.627, "cbam_direct_gei": 0.613, "target_gei": 0.612,
        "ebitda_cr": 17598.0, "eu_export_mt": 0.450, "bm_eu": 0.466
    },
    {
        "ticker": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Aluminum (Smelting)",
        "output_mt": 1.350, "scope1_mt": 25.200, "scope2_mt": 0.410,
        "actual_gei": 18.970, "cbam_direct_gei": 1.881, "target_gei": 18.322,
        "ebitda_cr": 38097.0, "eu_export_mt": 0.195, "bm_eu": 1.464
    },
    {
        "ticker": "SHREECEM.NS", "name": "Shree Cement", "sector": "Cement (60% Green)",
        "output_mt": 36.300, "scope1_mt": 22.031, "scope2_mt": 0.314,
        "actual_gei": 0.616, "cbam_direct_gei": 0.607, "target_gei": 0.611,
        "ebitda_cr": 4788.1, "eu_export_mt": 0.000, "bm_eu": 0.466
    }
]

# ----------------- SIMULATION CALCULATIONS -----------------
results = []
for c in cohort_data:
    # 1. Domestic CCTS Shortfall/Surplus
    delta_e = (c["actual_gei"] - c["target_gei"]) * c["output_mt"] * 1e6
    ccts_cost_cr = (delta_e * p_ccc) / 1e7 if delta_e > 0 else (delta_e * p_ccc * 0.7) / 1e7
    
    # 2. CBAM Tariff Liability
    cbam_gei = c["actual_gei"] if (c["ticker"] == "HINDALCO.NS" and indirect_expansion) else c["cbam_direct_gei"]
    net_taxable_intensity = max(0.0, cbam_gei - c["bm_eu"])
    
    dom_deduction = (max(0.0, c["actual_gei"] - c["target_gei"]) * p_ccc) if delta_art9 else 0.0
    cbam_rate_inr = max(0.0, (net_taxable_intensity * p_eu_ets * eur_inr) - dom_deduction)
    cbam_cost_cr = (c["eu_export_mt"] * 1e6 * cbam_rate_inr) / 1e7
    
    # 3. Aggregation & Margin Erosion
    total_liability_cr = max(0.0, ccts_cost_cr) + cbam_cost_cr
    ebitda_erosion_pct = (total_liability_cr / c["ebitda_cr"]) * 100
    
    results.append({
        "Ticker": c["ticker"],
        "Company": c["name"],
        "Sector": c["sector"],
        "Reported EBITDA (₹ Cr)": c["ebitda_cr"],
        "CCTS Liability (₹ Cr)": round(ccts_cost_cr, 2),
        "CBAM Tariff (₹ Cr)": round(cbam_cost_cr, 2),
        "Total Carbon Liability (₹ Cr)": round(total_liability_cr, 2),
        "EBITDA Erosion (%)": round(ebitda_erosion_pct, 2)
    })

df_res = pd.DataFrame(results)

# ----------------- TOP METRIC CARDS -----------------
col1, col2, col3 = st.columns(3)
basis_spread = (p_eu_ets * eur_inr) - p_ccc

col1.metric("EU-ETS vs CCTS Spread", f"₹{basis_spread:,.0f} / tCO2e", delta=f"{p_eu_ets} €/t ETS")
col2.metric("Portfolio Total Liability", f"₹{df_res['Total Carbon Liability (₹ Cr)'].sum():,.1f} Cr")
col3.metric("Peak Corporate Margin Drag", f"{df_res['EBITDA Erosion (%)'].max():.2f}%", delta=df_res.loc[df_res['EBITDA Erosion (%)'].idxmax()]['Company'], delta_color="inverse")

st.markdown("---")

# ----------------- SECTION 1: DATA TABLE -----------------
st.subheader("📋 Corporate Balance Sheet Impact Summary")
st.dataframe(df_res.style.highlight_max(subset=["EBITDA Erosion (%)"], color="#ffcdd2"), use_container_width=True)

# ----------------- SECTION 2: CHARTS -----------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Post-Carbon EBITDA Waterfall Bridge")
    selected_ticker = st.selectbox("Select Obligated Entity:", df_res["Ticker"].tolist())
    row = df_res[df_res["Ticker"] == selected_ticker].iloc[0]
    
    waterfall_fig = go.Figure(go.Waterfall(
        name="EBITDA Bridge",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Reported EBITDA", "CCTS Net Cost", "CBAM Tariff", "Adjusted EBITDA"],
        textposition="outside",
        text=[f"₹{row['Reported EBITDA (₹ Cr)']:,.0f}", f"-₹{max(0, row['CCTS Liability (₹ Cr)']):,.1f}", f"-₹{row['CBAM Tariff (₹ Cr)']:,.1f}", f"₹{row['Reported EBITDA (₹ Cr)'] - row['Total Carbon Liability (₹ Cr)']:,.0f}"],
        y=[row["Reported EBITDA (₹ Cr)"], -max(0, row["CCTS Liability (₹ Cr)"]), -row["CBAM Tariff (₹ Cr)"], 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef5350"}},
        totals={"marker": {"color": "#42a5f5"}}
    ))
    waterfall_fig.update_layout(title=f"{row['Company']} ({selected_ticker}) Margin Erosion", yaxis_title="₹ Crore", showlegend=False)
    st.plotly_chart(waterfall_fig, use_container_width=True)

with col_chart2:
    st.subheader("📉 Marginal Abatement Cost Curve (MACC)")
    # MACC engineering interventions
    macc_data = {
        "Technology": ["Waste Heat Recovery", "Captive Solar/Wind PPA", "Alternative Fuel (AFR)", "BF Top-Gas / PCI", "Scrap-EAF Route", "Green H2-DRI"],
        "Potential_Mt": [4.5, 8.2, 5.0, 7.1, 10.5, 12.0],
        "Cost_INR": [650, 950, 1150, 1550, 1850, 4200]
    }
    df_macc = pd.DataFrame(macc_data)
    
    macc_fig = go.Figure()
    macc_fig.add_trace(go.Bar(
        x=df_macc["Technology"],
        y=df_macc["Cost_INR"],
        marker_color=["#66bb6a" if c <= p_ccc else "#ffa726" for c in df_macc["Cost_INR"]],
        text=[f"₹{c}" for c in df_macc["Cost_INR"]],
        textposition="outside"
    ))
    macc_fig.add_hline(y=p_ccc, line_dash="dash", line_color="#d32f2f", annotation_text=f"Market CCC Price (₹{p_ccc})")
    macc_fig.update_layout(title="Abatement Capex vs. Market Buy Threshold", yaxis_title="Cost per tCO2e Abated (₹)", xaxis_title="Intervention")
    st.plotly_chart(macc_fig, use_container_width=True)