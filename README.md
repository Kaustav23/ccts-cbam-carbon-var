# Indian Carbon Market (CCTS) & EU-CBAM: Corporate Carbon-VaR Desk

Institutional-grade financial risk engine and empirical simulation modeling the earnings impact of the Indian Carbon Credit Trading Scheme (CCTS) and the European Carbon Border Adjustment Mechanism (EU-CBAM) on heavy industry majors (Steel, Cement, Aluminum).

---

## Executive Summary & Key Findings

1. **Cross-Border Policy Asymmetry**: Operational compliance under domestic CCTS does not hedge against export friction. Companies with net domestic surplus CCCs (e.g., JSW Steel) face over **₹1,200 Cr** in gross CBAM liabilities due to the EU's total embedded emission benchmark (BM_EU = 1.370 tCO2e/t).
2. **Statutory Penalty Cliff**: Under Section 26 of the Energy Conservation Act, credit shortfalls trigger a non-compliance penalty equal to twice the average market clearing price (2x P_CCC), leading to up to **26% EBITDA compression** for high-intensity coal-DRI producers under exchange illiquidity stress.
3. **MACC Arbitrage Threshold**: Capital expenditure on Waste Heat Recovery Systems (WHRS: ₹650/t) and captive renewable PPAs (₹950/t) yields positive NPV against market credit purchasing the moment domestic CCC prices exceed ₹1,000/t.
4. **Annex II Tail Risk**: Primary aluminum (Hindalco) maintains minimal exposure under direct Scope 1 rules, but faces an immediate **10.08% margin drawdown** if CBAM scope expands to captive coal electricity.

---

## Methodological Framework

- **Domestic CCTS Liability Mechanics**:
  Delta_E = (GEI_actual - GEI_target) * Output
  L_CCTS = max(0, Delta_E) * P_CCC + max(0, Delta_E_unsettled) * Penalty_2x

- **EU-CBAM Tariff Mechanics (Regulation EU 2023/956)**:
  Tariff_CBAM = Volume_export * max(0, GEI_direct - BM_EU) * (P_EU_ETS * FX_EUR_INR - delta_Art9 * P_CCC)

- **Stochastic Carbon-VaR Simulation**:
  Evaluated across 10,000 Monte Carlo paths using correlated Geometric Brownian Motion (GBM) for CCC prices and EU-ETS allowance pricing.

---

## Repository Structure

- ccts_cbam_quant_engine.py: Core Python mathematical engine, Monte Carlo VaR simulator, and SEBI BRSR Principle 6 data parser.
- pp.py: Streamlit interactive analytical dashboard with dynamic EBITDA waterfall bridges and MACC step-charts.
- ccts_cbam_simulation_output.txt: Full deterministic and stochastic engine log across base, moderate, and stress scenarios.
- equirements.txt: Python package dependencies.

---

## Local Installation & Execution

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the quantitative simulation
python ccts_cbam_quant_engine.py

# 3. Launch the interactive dashboard
streamlit run app.py
