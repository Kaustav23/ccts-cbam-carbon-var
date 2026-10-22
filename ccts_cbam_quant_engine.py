#!/usr/bin/env python3
"""
Institutional Quantitative Analytics Engine:
Evaluating Corporate Earnings at Risk (Carbon VaR) and Export Competitiveness
under India's Carbon Credit Trading Scheme (CCTS) and EU-CBAM.

Regulatory & Empirical Grounding:
- Bureau of Energy Efficiency (BEE) & MoEFCC Greenhouse Gas Emissions Intensity (GEI)
  Target Rules, 2025/2026 under the Energy Conservation Act, 2001 (2022 Amendment).
- Central Electricity Regulatory Commission (CERC) (Terms and Conditions for Purchase
  and Sale of Carbon Credit Certificates) Regulations, 2026 (Notified Feb 27, 2026).
- EU Regulation (EU) 2023/956 establishing a Carbon Border Adjustment Mechanism (CBAM),
  specifically Article 9 (Carbon price paid in a third country), Annex II scope rules,
  and Implementing Regulation (EU) 2023/1773 (installation-level vs corporate-level
  boundary reconciliation for captive thermal power in primary aluminum smelting).
- Empirical FY2024-25 & FY2025-26 SEBI Business Responsibility and Sustainability Reports
  (BRSR - Principle 6) and Audited Financial Filings for Top 5 NSE-Listed Heavy Emitters
  (+ Shree Cement sector comparator):
  1. Tata Steel Ltd. (TATASTEEL.NS)
  2. JSW Steel Ltd. (JSWSTEEL.NS)
  3. Jindal Steel Ltd. / JSPL (JINDALSTEL.NS)
  4. UltraTech Cement Ltd. (ULTRACEMCO.NS)
  5. Hindalco Industries Ltd. (HINDALCO.NS)
  6. Shree Cement Ltd. (SHREECEM.NS)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# PHASE 2: DATA EXTRACTION & STRUCTURING SCHEMA (TIDY PANDAS DATAFRAME LAYOUT)
# ==============================================================================

@dataclass(frozen=True)
class BRSRComplianceRecord:
    """
    Strict schema representing a single corporate reporting entity's SEBI BRSR
    Principle 6 emissions disclosure, financial operating metrics, and CBAM exposure.
    """
    ticker: str
    company_name: str
    reporting_year: str
    sector: str                                  # 'Steel', 'Cement', 'Aluminum'
    production_route: str                        # e.g., 'BF-BOF / DRI', 'Clinker-Grey Cement', 'Coal-Captive Smelting'
    production_volume_t: float                   # Crude steel / Cementitious / Primary Aluminum output (metric tons)
    scope1_tco2e: float                          # Direct emissions (Scope 1, tCO2e - organizational boundary)
    scope2_tco2e: float                          # Indirect energy emissions (Scope 2, tCO2e - grid imports)
    cbam_direct_process_gei_tco2e_per_t: float   # EU-CBAM Annex II installation-level direct process GEI (excludes captive power)
    baseline_gei_fy24_tco2e_per_t: float         # BEE FY2023-24 verified baseline GEI (tCO2e / ton product)
    statutory_reduction_target_pct: float        # BEE 2-year cumulative reduction target (%)
    year1_compliance_weight: float               # Back-loaded trajectory weight (0.40 in Year 1 FY26, 1.00 in Year 2 FY27)
    revenue_inr_cr: float                        # Consolidated/India Revenue from Operations (INR Crore)
    ebitda_inr_cr: float                         # Operating EBITDA (INR Crore)
    pat_inr_cr: float                            # Profit After Tax (INR Crore)
    eu_export_revenue_pct: float                 # Share of total revenue derived from EU exports (%)
    eu_export_volume_t: float                    # Physical volume exported to EU under CBAM CN codes (metric tons)
    cbam_annex2_direct_only: bool                # True for Steel/Aluminum (Annex II: direct only), False for Cement
    cbam_eu_benchmark_tco2e_per_t: float         # EU ETS / CBAM route benchmark (e.g., 1.370 tCO2e/t for BF-BOF steel)
    macc_marginal_cost_inr_per_tco2e: float      # Marginal Abatement Cost Curve (MACC) weighted technology cost (INR/tCO2e)
    abate_capex_per_annual_tco2e_inr: float      # Upfront Capex intensity per annual tCO2e abated (INR / tCO2e / yr)
    wacc_pct: float                              # Corporate Weighted Average Cost of Capital (%)


class BRSRDataIngestEngine:
    """
    Ingests, validates, and structures empirical SEBI BRSR Principle 6 disclosures
    and annual financial statements into a standardized tidy pandas DataFrame.
    """

    REQUIRED_COLUMNS: List[str] = [
        "ticker",
        "company_name",
        "reporting_year",
        "sector",
        "production_route",
        "production_volume_t",
        "scope1_tco2e",
        "scope2_tco2e",
        "total_scope12_tco2e",
        "actual_gei_tco2e_per_t",
        "direct_gei_tco2e_per_t",
        "cbam_direct_process_gei_tco2e_per_t",
        "baseline_gei_fy24_tco2e_per_t",
        "statutory_gei_target_y1_tco2e_per_t",
        "statutory_gei_target_y2_tco2e_per_t",
        "revenue_inr_cr",
        "ebitda_inr_cr",
        "pat_inr_cr",
        "ebitda_margin_pct",
        "net_margin_pct",
        "eu_export_revenue_pct",
        "eu_export_volume_t",
        "cbam_annex2_direct_only",
        "cbam_eu_benchmark_tco2e_per_t",
        "macc_marginal_cost_inr_per_tco2e",
        "abate_capex_per_annual_tco2e_inr",
        "wacc_pct",
    ]

    @staticmethod
    def load_empirical_nse_cohort() -> pd.DataFrame:
        """
        Loads empirical FY2025-26 / FY2024-25 SEBI BRSR Principle 6 and Audited
        Financial metrics for the Top 5 NSE-listed heavy emitters (+ Shree Cement peer).
        """
        records: List[BRSRComplianceRecord] = [
            # 1. Tata Steel Ltd. (India Gate-to-Gate CCTS Perimeter + Consolidated Financials)
            BRSRComplianceRecord(
                ticker="TATASTEEL.NS",
                company_name="Tata Steel Limited",
                reporting_year="FY2025-26",
                sector="Steel",
                production_route="BF-BOF (Integrated)",
                production_volume_t=23_430_000.0,
                scope1_tco2e=54_100_000.0,
                scope2_tco2e=2_200_000.0,
                cbam_direct_process_gei_tco2e_per_t=2.309,  # 54.10M / 23.43M tcs direct BF-BOF intensity
                baseline_gei_fy24_tco2e_per_t=2.440,
                statutory_reduction_target_pct=5.20,
                year1_compliance_weight=0.40,
                revenue_inr_cr=232_140.0,
                ebitda_inr_cr=34_848.0,
                pat_inr_cr=10_886.0,
                eu_export_revenue_pct=5.80,
                eu_export_volume_t=1_350_000.0,
                cbam_annex2_direct_only=True,
                cbam_eu_benchmark_tco2e_per_t=1.370,
                macc_marginal_cost_inr_per_tco2e=1_650.0,
                abate_capex_per_annual_tco2e_inr=11_500.0,
                wacc_pct=10.50,
            ),
            # 2. JSW Steel Ltd.
            BRSRComplianceRecord(
                ticker="JSWSTEEL.NS",
                company_name="JSW Steel Limited",
                reporting_year="FY2025-26",
                sector="Steel",
                production_route="BF-BOF & Corex/DRI",
                production_volume_t=26_850_000.0,
                scope1_tco2e=60_258_642.0,
                scope2_tco2e=1_022_973.0,
                cbam_direct_process_gei_tco2e_per_t=2.244,  # 60.26M / 26.85M tcs direct process intensity
                baseline_gei_fy24_tco2e_per_t=2.360,
                statutory_reduction_target_pct=5.50,
                year1_compliance_weight=0.40,
                revenue_inr_cr=185_470.0,
                ebitda_inr_cr=29_821.0,
                pat_inr_cr=25_508.0,
                eu_export_revenue_pct=8.40,
                eu_export_volume_t=2_150_000.0,
                cbam_annex2_direct_only=True,
                cbam_eu_benchmark_tco2e_per_t=1.370,
                macc_marginal_cost_inr_per_tco2e=1_520.0,
                abate_capex_per_annual_tco2e_inr=10_800.0,
                wacc_pct=10.80,
            ),
            # 3. Jindal Steel Limited (formerly Jindal Steel & Power Ltd.)
            BRSRComplianceRecord(
                ticker="JINDALSTEL.NS",
                company_name="Jindal Steel Limited",
                reporting_year="FY2025-26",
                sector="Steel",
                production_route="Coal-DRI & BF-BOF",
                production_volume_t=12_850_000.0,
                scope1_tco2e=35_630_000.0,
                scope2_tco2e=690_000.0,
                cbam_direct_process_gei_tco2e_per_t=2.773,  # 35.63M / 12.85M tcs direct coal-DRI + BF-BOF intensity
                baseline_gei_fy24_tco2e_per_t=2.790,
                statutory_reduction_target_pct=6.00,
                year1_compliance_weight=0.40,
                revenue_inr_cr=62_412.0,
                ebitda_inr_cr=9_099.0,
                pat_inr_cr=3_361.0,
                eu_export_revenue_pct=4.50,
                eu_export_volume_t=420_000.0,
                cbam_annex2_direct_only=True,
                cbam_eu_benchmark_tco2e_per_t=1.370,
                macc_marginal_cost_inr_per_tco2e=1_480.0,
                abate_capex_per_annual_tco2e_inr=10_200.0,
                wacc_pct=11.20,
            ),
            # 4. UltraTech Cement Limited
            BRSRComplianceRecord(
                ticker="ULTRACEMCO.NS",
                company_name="UltraTech Cement Limited",
                reporting_year="FY2025-26",
                sector="Cement",
                production_route="Dry Process Clinker & Blended Cement",
                production_volume_t=145_000_000.0,
                scope1_tco2e=88_884_246.0,
                scope2_tco2e=2_040_172.0,
                cbam_direct_process_gei_tco2e_per_t=0.613,  # Calcination + kiln fuel direct intensity
                baseline_gei_fy24_tco2e_per_t=0.648,
                statutory_reduction_target_pct=5.60,
                year1_compliance_weight=0.40,
                revenue_inr_cr=87_384.0,
                ebitda_inr_cr=17_598.0,
                pat_inr_cr=8_305.0,
                eu_export_revenue_pct=0.35,
                eu_export_volume_t=450_000.0,
                cbam_annex2_direct_only=False,              # Cement includes direct + indirect under CBAM
                cbam_eu_benchmark_tco2e_per_t=0.466,
                macc_marginal_cost_inr_per_tco2e=980.0,
                abate_capex_per_annual_tco2e_inr=7_400.0,
                wacc_pct=9.80,
            ),
            # 5. Hindalco Industries Limited
            # Note on Boundary Reconciliation:
            # Organizational BRSR Scope 1 (25.20M tCO2e) includes on-site captive coal thermal power (~22.66M tCO2e)
            # plus Hall-Heroult potline anode/PFC direct process emissions (~2.54M tCO2e = 1.881 tCO2e/t Al).
            # Under BEE CCTS (gate-to-gate), captive power is INCLUDED in GEI (18.970 tCO2e/t Al).
            # Under EU-CBAM Annex II (Implementing Reg 2023/1773), electricity from captive power is classified as
            # INDIRECT emissions; only potline process emissions (1.881 tCO2e/t Al vs 1.464 benchmark) are taxed
            # unless indirect scope expansion is triggered!
            BRSRComplianceRecord(
                ticker="HINDALCO.NS",
                company_name="Hindalco Industries Limited",
                reporting_year="FY2025-26",
                sector="Aluminum",
                production_route="Hall-Heroult Smelting (Captive Coal Power)",
                production_volume_t=1_350_000.0,
                scope1_tco2e=25_200_000.0,
                scope2_tco2e=410_000.0,
                cbam_direct_process_gei_tco2e_per_t=1.881,  # Potline anode oxidation + PFC direct emissions (Annex II)
                baseline_gei_fy24_tco2e_per_t=19.450,
                statutory_reduction_target_pct=5.80,
                year1_compliance_weight=0.40,
                revenue_inr_cr=274_944.0,
                ebitda_inr_cr=38_097.0,
                pat_inr_cr=13_391.0,
                eu_export_revenue_pct=3.20,
                eu_export_volume_t=195_000.0,
                cbam_annex2_direct_only=True,
                cbam_eu_benchmark_tco2e_per_t=1.464,
                macc_marginal_cost_inr_per_tco2e=1_890.0,
                abate_capex_per_annual_tco2e_inr=13_600.0,
                wacc_pct=10.40,
            ),
            # 6. Shree Cement Limited (Cement Sector Low-GEI Benchmark Peer)
            BRSRComplianceRecord(
                ticker="SHREECEM.NS",
                company_name="Shree Cement Limited",
                reporting_year="FY2025-26",
                sector="Cement",
                production_route="High-WHRS (60% Green Power) Blended Cement",
                production_volume_t=36_300_000.0,
                scope1_tco2e=22_031_435.0,
                scope2_tco2e=313_672.0,
                cbam_direct_process_gei_tco2e_per_t=0.607,
                baseline_gei_fy24_tco2e_per_t=0.642,
                statutory_reduction_target_pct=4.90,
                year1_compliance_weight=0.40,
                revenue_inr_cr=19_310.5,
                ebitda_inr_cr=4_788.1,
                pat_inr_cr=1_706.3,
                eu_export_revenue_pct=0.00,
                eu_export_volume_t=0.0,
                cbam_annex2_direct_only=False,
                cbam_eu_benchmark_tco2e_per_t=0.466,
                macc_marginal_cost_inr_per_tco2e=920.0,
                abate_capex_per_annual_tco2e_inr=7_100.0,
                wacc_pct=9.60,
            ),
        ]

        df = pd.DataFrame([asdict(r) for r in records])

        # Derived Physical & Financial Intensity Columns
        df["total_scope12_tco2e"] = df["scope1_tco2e"] + df["scope2_tco2e"]
        df["actual_gei_tco2e_per_t"] = df["total_scope12_tco2e"] / df["production_volume_t"]
        df["direct_gei_tco2e_per_t"] = df["scope1_tco2e"] / df["production_volume_t"]

        # Back-loaded BEE GEI Targets:
        # Year 1 (FY2025-26): 40% of two-year reduction target
        # Year 2 (FY2026-27): 100% of cumulative two-year reduction target (40% Y1 + 60% Y2)
        df["statutory_gei_target_y1_tco2e_per_t"] = df["baseline_gei_fy24_tco2e_per_t"] * (
            1.0 - (df["statutory_reduction_target_pct"] / 100.0) * df["year1_compliance_weight"]
        )
        df["statutory_gei_target_y2_tco2e_per_t"] = df["baseline_gei_fy24_tco2e_per_t"] * (
            1.0 - (df["statutory_reduction_target_pct"] / 100.0)
        )

        df["ebitda_margin_pct"] = (df["ebitda_inr_cr"] / df["revenue_inr_cr"]) * 100.0
        df["net_margin_pct"] = (df["pat_inr_cr"] / df["revenue_inr_cr"]) * 100.0

        # Validate Schema
        missing = set(BRSRDataIngestEngine.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"BRSR schema validation failed; missing columns: {missing}")

        return df[BRSRDataIngestEngine.REQUIRED_COLUMNS].copy()


# ==============================================================================
# PHASE 1 & PHASE 3: QUANTITATIVE ENGINE (CCTS + EU-CBAM + MONTE CARLO VaR)
# ==============================================================================

@dataclass
class MarketScenarioParams:
    """
    Parameters governing a deterministic or stochastic compliance scenario.
    """
    name: str
    ccc_price_inr_per_t: float                 # Domestic CCC market clearing price (Rs / tCO2e)
    eu_ets_price_eur_per_t: float              # EU-ETS allowance price (EUR / tCO2e)
    eur_inr_fx: float                          # EUR to INR exchange rate
    compliance_year_stage: int                 # 1 = FY2025-26 (40% target), 2 = FY2026-27 (100% target)
    custom_reduction_override_pct: Optional[float] = None  # Optional override for sensitivity matrix (2% - 10%)
    exchange_procurement_ratio: float = 0.85   # Share of shortfall covered via Power Exchange CCC purchase (theta_i)
                                               # Remaining (1 - theta_i) faces 2x average CCC price penalty under EC Act Sec 26
    surplus_monetization_ratio: float = 0.75   # Share of earned surplus CCCs sold vs banked
    cbam_phase_in_factor: float = 0.025        # 2.5% in 2026 (97.5% free allocation); 0.485 in 2030; 1.00 in 2034
    cbam_article9_recognized: bool = False     # False = current status (CCTS not eligible for Art 9 offset)
    cbam_include_captive_indirect: bool = False  # True = EU expands Annex II to include captive power indirect emissions


class CCTSCBAMQuantEngine:
    """
    Quantitative risk engine computing:
    1. Carbon Credit Shortfall/Surplus (Delta E in tCO2e)
    2. Compliance Financial Liability (L_compliance in INR Crore) including 2x CCC statutory penalty
    3. Net Export CBAM Friction (T_CBAM in INR Crore) under EU Reg 2023/956 Article 9
    4. Carbon Value at Risk (Carbon-VaR & CVaR on EBITDA and Operating Margins)
    5. Marginal Abatement Cost Curve (MACC) Breakeven Decarbonization Capex
    """

    FIXED_PENALTY_INR_CR: float = 0.10  # Up to Rs 10 Lakh (0.10 Cr) fixed fine under Sec 26 Energy Conservation Act

    def __init__(self, df_cohort: pd.DataFrame):
        self.df = df_cohort.copy()

    def evaluate_scenario(self, params: MarketScenarioParams) -> pd.DataFrame:
        """
        Evaluates all equations deterministically for a given MarketScenarioParams configuration.
        """
        res = self.df.copy()
        res["scenario_name"] = params.name
        res["ccc_price_inr"] = params.ccc_price_inr_per_t
        res["eu_ets_price_eur"] = params.eu_ets_price_eur_per_t
        res["eu_ets_price_inr"] = params.eu_ets_price_eur_per_t * params.eur_inr_fx

        # ----------------------------------------------------------------------
        # Equation 1: Carbon Credit Shortfall / Surplus (\Delta E_i in tCO2e)
        # \Delta E_i = (I_i^{act} - I_i^*) * Q_i
        # Positive \Delta E_i = Shortfall (Deficit Buyer); Negative \Delta E_i = Surplus (Credit Seller)
        # ----------------------------------------------------------------------
        if params.custom_reduction_override_pct is not None:
            res["target_gei_tco2e_per_t"] = res["baseline_gei_fy24_tco2e_per_t"] * (
                1.0 - (params.custom_reduction_override_pct / 100.0)
            )
        elif params.compliance_year_stage == 1:
            res["target_gei_tco2e_per_t"] = res["statutory_gei_target_y1_tco2e_per_t"]
        else:
            res["target_gei_tco2e_per_t"] = res["statutory_gei_target_y2_tco2e_per_t"]

        res["gei_gap_tco2e_per_t"] = res["actual_gei_tco2e_per_t"] - res["target_gei_tco2e_per_t"]
        res["delta_e_tco2e"] = res["gei_gap_tco2e_per_t"] * res["production_volume_t"]
        res["compliance_status"] = np.where(res["delta_e_tco2e"] > 0, "Shortfall (Deficit)", "Surplus (Credit Earner)")

        # ----------------------------------------------------------------------
        # Equation 2: Domestic CCTS Compliance Financial Liability (L_compliance)
        # For \Delta E_i > 0:
        #   L_compliance = \Delta E_i * [ \theta * P_CCC + (1 - \theta) * (2 * P_CCC) ] + F_fixed
        # For \Delta E_i <= 0:
        #   L_compliance = \gamma_mon * \Delta E_i * P_CCC (negative liability = revenue inflow)
        # ----------------------------------------------------------------------
        theta = params.exchange_procurement_ratio
        effective_shortfall_unit_price = params.ccc_price_inr_per_t * (theta + 2.0 * (1.0 - theta))

        shortfall_mask = res["delta_e_tco2e"] > 0
        fixed_penalty_cr = np.where(shortfall_mask & (theta < 1.0), self.FIXED_PENALTY_INR_CR, 0.0)

        res["ccts_exchange_cost_inr_cr"] = np.where(
            shortfall_mask,
            (res["delta_e_tco2e"] * theta * params.ccc_price_inr_per_t) / 1e7,
            (res["delta_e_tco2e"] * params.surplus_monetization_ratio * params.ccc_price_inr_per_t) / 1e7,
        )
        res["ccts_penalty_2x_inr_cr"] = np.where(
            shortfall_mask,
            (res["delta_e_tco2e"] * (1.0 - theta) * 2.0 * params.ccc_price_inr_per_t) / 1e7 + fixed_penalty_cr,
            0.0,
        )
        res["l_compliance_inr_cr"] = res["ccts_exchange_cost_inr_cr"] + res["ccts_penalty_2x_inr_cr"]

        # ----------------------------------------------------------------------
        # Equation 3: Cross-Border EU-CBAM Friction (T_CBAM) under Reg 2023/956 Art 9
        # Specific embedded emissions e_{i,CBAM}:
        #   If cbam_include_captive_indirect is True -> full gate-to-gate actual_gei_tco2e_per_t
        #   Else if Steel & Aluminum (Annex II) -> cbam_direct_process_gei_tco2e_per_t
        #   Else (Cement) -> actual_gei_tco2e_per_t (Direct + Indirect)
        # Net taxable specific emissions after EU ETS free allocation adjustment:
        #   e_{i,net} = max(0, e_{i,CBAM} - (1 - f_CBAM) * B_{i,EU})
        # Effective domestic carbon price paid per ton of product:
        #   \tau_{i,dom} = max(0, L_compliance_INR) / Q_i
        # T_CBAM = Q_i^{EU} * max(0, e_{i,net} * P_{EU-ETS,INR} - \delta_{Art9} * \tau_{i,dom})
        # ----------------------------------------------------------------------
        if params.cbam_include_captive_indirect:
            res["cbam_specific_emissions_tco2e_per_t"] = res["actual_gei_tco2e_per_t"]
        else:
            res["cbam_specific_emissions_tco2e_per_t"] = np.where(
                res["cbam_annex2_direct_only"],
                res["cbam_direct_process_gei_tco2e_per_t"],
                res["actual_gei_tco2e_per_t"],
            )

        free_alloc_multiplier = 1.0 - params.cbam_phase_in_factor
        res["cbam_net_taxable_gei_tco2e_per_t"] = np.maximum(
            0.0,
            res["cbam_specific_emissions_tco2e_per_t"] - free_alloc_multiplier * res["cbam_eu_benchmark_tco2e_per_t"],
        )

        # Basis spread per tCO2e between EU-ETS and Indian CCC
        res["carbon_basis_spread_inr_per_tco2e"] = res["eu_ets_price_inr"] - params.ccc_price_inr_per_t

        # Gross CBAM charge per ton of exported product (INR / ton product)
        res["gross_cbam_tariff_inr_per_t_product"] = (
            res["cbam_net_taxable_gei_tco2e_per_t"] * res["eu_ets_price_inr"]
        )

        # Domestic carbon cost effectively paid per ton of physical output (INR / ton product)
        res["domestic_carbon_cost_inr_per_t_product"] = (
            np.maximum(0.0, res["ccts_exchange_cost_inr_cr"] * 1e7) / res["production_volume_t"]
        )

        art9_deduction_inr_per_t = np.where(
            params.cbam_article9_recognized,
            res["domestic_carbon_cost_inr_per_t_product"],
            0.0,
        )

        res["net_cbam_tariff_inr_per_t_product"] = np.maximum(
            0.0,
            res["gross_cbam_tariff_inr_per_t_product"] - art9_deduction_inr_per_t,
        )

        res["t_cbam_inr_cr"] = (res["eu_export_volume_t"] * res["net_cbam_tariff_inr_per_t_product"]) / 1e7

        # ----------------------------------------------------------------------
        # Equation 4: Total Carbon Liability, EBITDA VaR, Margin Compression & Breakeven Capex
        # ----------------------------------------------------------------------
        res["total_carbon_liability_inr_cr"] = res["l_compliance_inr_cr"] + res["t_cbam_inr_cr"]
        res["carbon_var_ebitda_pct"] = (res["total_carbon_liability_inr_cr"] / res["ebitda_inr_cr"]) * 100.0
        res["margin_erosion_bps"] = (res["total_carbon_liability_inr_cr"] / res["revenue_inr_cr"]) * 10000.0

        # Post-Carbon Adjusted EBITDA & Margin
        res["adjusted_ebitda_inr_cr"] = res["ebitda_inr_cr"] - res["total_carbon_liability_inr_cr"]
        res["adjusted_ebitda_margin_pct"] = (res["adjusted_ebitda_inr_cr"] / res["revenue_inr_cr"]) * 100.0

        # Breakeven Decarbonization Capex:
        r = res["wacc_pct"] / 100.0
        annuity_factor_10yr = (1.0 - (1.0 + r) ** (-10)) / r
        res["engineering_decarb_capex_inr_cr"] = (
            np.maximum(0.0, res["delta_e_tco2e"]) * res["abate_capex_per_annual_tco2e_inr"]
        ) / 1e7
        res["npv_breakeven_capex_ceiling_inr_cr"] = (
            np.maximum(0.0, res["total_carbon_liability_inr_cr"]) * annuity_factor_10yr
        )
        res["macc_abate_vs_buy_signal"] = np.where(
            res["macc_marginal_cost_inr_per_tco2e"] <= effective_shortfall_unit_price,
            "ABATE (Internal Capex < Effective Carbon Cost)",
            "BUY CCCs (Market Credit < MACC)",
        )

        return res

    def run_sensitivity_matrix(
        self,
        ccc_prices: List[float] = [500.0, 1000.0, 1200.0, 1800.0, 2500.0, 3000.0],
        reduction_targets_pct: List[float] = [2.0, 4.0, 6.0, 8.0, 10.0],
        eu_ets_eur: float = 85.0,
        eur_inr_fx: float = 92.0,
        cbam_phase_in_factor: float = 0.025,
        exchange_procurement_ratio: float = 0.85,
    ) -> pd.DataFrame:
        """
        Generates the full 2D deterministic sensitivity matrix across CCC prices (Rs 500 - Rs 3,000/t)
        and statutory GEI reduction targets (2% - 10%) for every company in the cohort.
        """
        rows = []
        for p_ccc in ccc_prices:
            for red_pct in reduction_targets_pct:
                params = MarketScenarioParams(
                    name=f"P_CCC={int(p_ccc)}_Target={red_pct:.1f}%",
                    ccc_price_inr_per_t=p_ccc,
                    eu_ets_price_eur_per_t=eu_ets_eur,
                    eur_inr_fx=eur_inr_fx,
                    compliance_year_stage=2,
                    custom_reduction_override_pct=red_pct,
                    exchange_procurement_ratio=exchange_procurement_ratio,
                    cbam_phase_in_factor=cbam_phase_in_factor,
                    cbam_article9_recognized=False,
                    cbam_include_captive_indirect=False,
                )
                df_scen = self.evaluate_scenario(params)
                df_scen["matrix_ccc_price_inr"] = p_ccc
                df_scen["matrix_reduction_target_pct"] = red_pct
                rows.append(df_scen)

        return pd.concat(rows, ignore_index=True)

    def run_monte_carlo_carbon_var(
        self,
        n_simulations: int = 10_000,
        seed: int = 42,
        cbam_phase_in_factor: float = 0.025,
    ) -> pd.DataFrame:
        """
        Runs a 10,000-path Monte Carlo simulation modeling joint stochastic shocks across:
        - Domestic CCC price (Triangular spanning Rs 500 to Rs 3,000/t, mode Rs 1,200/t)
        - EU-ETS allowance price (Normal truncated spanning EUR 70 to EUR 100/t, mean EUR 85/t)
        - EUR/INR FX rate (Normal around 92.0 with sigma = 2.2)
        - Regulatory GEI target stringency (Uniform spanning 2% to 10%)
        - Market liquidity shortfall / penalty exposure (1 - theta_i ~ Beta(1.8, 8.2))
        Outputs Mean, Median, 95% Carbon-VaR, and 99% Expected Shortfall (CVaR) per company.
        """
        rng = np.random.default_rng(seed)

        p_ccc_sim = rng.triangular(left=500.0, mode=1200.0, right=3000.0, size=n_simulations)
        p_eu_sim = np.clip(rng.normal(loc=85.0, scale=8.5, size=n_simulations), 70.0, 100.0)
        fx_sim = np.clip(rng.normal(loc=92.0, scale=2.2, size=n_simulations), 86.0, 100.0)
        alpha_sim = rng.uniform(low=0.02, high=0.10, size=n_simulations)
        penalty_share_sim = rng.beta(a=1.8, b=8.2, size=n_simulations)

        mc_results = []
        for _, row in self.df.iterrows():
            q_prod = row["production_volume_t"]
            i_act = row["actual_gei_tco2e_per_t"]
            i_base = row["baseline_gei_fy24_tco2e_per_t"]

            i_star = i_base * (1.0 - alpha_sim)
            delta_e = (i_act - i_star) * q_prod

            unit_eff_price = p_ccc_sim * ((1.0 - penalty_share_sim) + 2.0 * penalty_share_sim)
            l_comp_cr = np.where(
                delta_e > 0.0,
                (delta_e * unit_eff_price) / 1e7 + self.FIXED_PENALTY_INR_CR,
                (delta_e * 0.75 * p_ccc_sim) / 1e7,
            )

            e_cbam = (
                row["cbam_direct_process_gei_tco2e_per_t"]
                if row["cbam_annex2_direct_only"]
                else i_act
            )
            e_net = max(0.0, e_cbam - (1.0 - cbam_phase_in_factor) * row["cbam_eu_benchmark_tco2e_per_t"])
            t_cbam_cr = (row["eu_export_volume_t"] * e_net * p_eu_sim * fx_sim) / 1e7

            total_loss_cr = l_comp_cr + t_cbam_cr
            ebitda_erosion_pct = (total_loss_cr / row["ebitda_inr_cr"]) * 100.0
            margin_erosion_bps = (total_loss_cr / row["revenue_inr_cr"]) * 10000.0

            var_95_ebitda_pct = float(np.percentile(ebitda_erosion_pct, 95))
            var_99_ebitda_pct = float(np.percentile(ebitda_erosion_pct, 99))
            cvar_95_ebitda_pct = float(ebitda_erosion_pct[ebitda_erosion_pct >= var_95_ebitda_pct].mean())
            cvar_99_ebitda_pct = float(ebitda_erosion_pct[ebitda_erosion_pct >= var_99_ebitda_pct].mean())

            var_95_inr_cr = float(np.percentile(total_loss_cr, 95))
            cvar_95_inr_cr = float(total_loss_cr[total_loss_cr >= var_95_inr_cr].mean())

            mc_results.append({
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "sector": row["sector"],
                "mean_loss_inr_cr": float(np.mean(total_loss_cr)),
                "median_loss_inr_cr": float(np.median(total_loss_cr)),
                "std_loss_inr_cr": float(np.std(total_loss_cr)),
                "var_95_loss_inr_cr": var_95_inr_cr,
                "cvar_95_loss_inr_cr": cvar_95_inr_cr,
                "mean_ebitda_erosion_pct": float(np.mean(ebitda_erosion_pct)),
                "var_95_ebitda_pct": var_95_ebitda_pct,
                "cvar_95_ebitda_pct": cvar_95_ebitda_pct,
                "var_99_ebitda_pct": var_99_ebitda_pct,
                "cvar_99_ebitda_pct": cvar_99_ebitda_pct,
                "var_95_margin_erosion_bps": float(np.percentile(margin_erosion_bps, 95)),
                "prob_shortfall_pct": float(np.mean(delta_e > 0.0) * 100.0),
            })

        return pd.DataFrame(mc_results)


def main() -> None:
    """
    Executes the full quantitative pipeline and prints verified tables for inspection.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    df_cohort = BRSRDataIngestEngine.load_empirical_nse_cohort()
    engine = CCTSCBAMQuantEngine(df_cohort)

    print("=" * 100)
    print("1. EMPIRICAL SEBI BRSR PRINCIPLE 6 & FINANCIAL DATASET (FY2025-26 COHORT)")
    print("=" * 100)
    print(df_cohort[[
        "ticker", "sector", "production_volume_t", "scope1_tco2e", "scope2_tco2e",
        "actual_gei_tco2e_per_t", "cbam_direct_process_gei_tco2e_per_t", "baseline_gei_fy24_tco2e_per_t",
        "statutory_gei_target_y1_tco2e_per_t", "statutory_gei_target_y2_tco2e_per_t",
        "revenue_inr_cr", "ebitda_inr_cr", "ebitda_margin_pct", "eu_export_revenue_pct", "eu_export_volume_t"
    ]])

    scenarios = [
        MarketScenarioParams(
            name="Base (P_CCC=₹500, EU=€70, Y1 40% Target, 2026 CBAM 2.5% Annex II)",
            ccc_price_inr_per_t=500.0,
            eu_ets_price_eur_per_t=70.0,
            eur_inr_fx=92.0,
            compliance_year_stage=1,
            exchange_procurement_ratio=1.00,
            cbam_phase_in_factor=0.025,
            cbam_article9_recognized=False,
            cbam_include_captive_indirect=False,
        ),
        MarketScenarioParams(
            name="Medium (P_CCC=₹1,200, EU=€85, Y2 100% Target, 2026 CBAM 2.5% Annex II)",
            ccc_price_inr_per_t=1200.0,
            eu_ets_price_eur_per_t=85.0,
            eur_inr_fx=92.0,
            compliance_year_stage=2,
            exchange_procurement_ratio=0.85,
            cbam_phase_in_factor=0.025,
            cbam_article9_recognized=False,
            cbam_include_captive_indirect=False,
        ),
        MarketScenarioParams(
            name="Stress (P_CCC=₹2,500, EU=€100, Y2 100% Target, 100% 2x Penalty + CBAM Full Annex II)",
            ccc_price_inr_per_t=2500.0,
            eu_ets_price_eur_per_t=100.0,
            eur_inr_fx=92.0,
            compliance_year_stage=2,
            exchange_procurement_ratio=0.00,
            cbam_phase_in_factor=1.000,
            cbam_article9_recognized=False,
            cbam_include_captive_indirect=False,
        ),
        MarketScenarioParams(
            name="Stress_Captive_Indirect (P_CCC=₹2,500, EU=€100, Y2 Target, 2x Penalty + CBAM Indirect Scope Expansion)",
            ccc_price_inr_per_t=2500.0,
            eu_ets_price_eur_per_t=100.0,
            eur_inr_fx=92.0,
            compliance_year_stage=2,
            exchange_procurement_ratio=0.00,
            cbam_phase_in_factor=1.000,
            cbam_article9_recognized=False,
            cbam_include_captive_indirect=True,
        ),
        MarketScenarioParams(
            name="Stress_With_Art9 (P_CCC=₹2,500, EU=€100, Y2 Target, CBAM Full Annex II + Article 9 Recognized)",
            ccc_price_inr_per_t=2500.0,
            eu_ets_price_eur_per_t=100.0,
            eur_inr_fx=92.0,
            compliance_year_stage=2,
            exchange_procurement_ratio=1.00,
            cbam_phase_in_factor=1.000,
            cbam_article9_recognized=True,
            cbam_include_captive_indirect=False,
        ),
    ]

    for sc in scenarios:
        print("\n" + "=" * 100)
        print(f"SCENARIO: {sc.name}")
        print("=" * 100)
        df_sc = engine.evaluate_scenario(sc)
        print(df_sc[[
            "ticker", "actual_gei_tco2e_per_t", "target_gei_tco2e_per_t", "delta_e_tco2e",
            "ccts_exchange_cost_inr_cr", "ccts_penalty_2x_inr_cr", "l_compliance_inr_cr",
            "t_cbam_inr_cr", "total_carbon_liability_inr_cr", "carbon_var_ebitda_pct",
            "margin_erosion_bps", "engineering_decarb_capex_inr_cr", "npv_breakeven_capex_ceiling_inr_cr",
            "macc_abate_vs_buy_signal"
        ]])

    print("\n" + "=" * 100)
    print("3. MONTE CARLO SIMULATION (10,000 PATHS, 2026 CBAM PHASE-IN 2.5% ANNEX II)")
    print("=" * 100)
    df_mc_2026 = engine.run_monte_carlo_carbon_var(n_simulations=10_000, seed=42, cbam_phase_in_factor=0.025)
    print(df_mc_2026)

    print("\n" + "=" * 100)
    print("4. MONTE CARLO SIMULATION (10,000 PATHS, FULL CBAM EXPOSURE / UNPHASED BENCHMARK)")
    print("=" * 100)
    df_mc_full = engine.run_monte_carlo_carbon_var(n_simulations=10_000, seed=42, cbam_phase_in_factor=1.000)
    print(df_mc_full)

    print("\n" + "=" * 100)
    print("5. DETERMINISTIC SENSITIVITY MATRIX (EBITDA EROSION % AT 2026 CBAM 2.5% ANNEX II, THETA=0.85)")
    print("=" * 100)
    df_mat = engine.run_sensitivity_matrix(cbam_phase_in_factor=0.025, exchange_procurement_ratio=0.85)
    piv = df_mat.pivot_table(
        index=["ticker", "matrix_reduction_target_pct"],
        columns="matrix_ccc_price_inr",
        values="carbon_var_ebitda_pct",
    )
    print(piv)


if __name__ == "__main__":
    main()

# Save deterministic and stress results to CSV
try:
    df_base.to_csv("ccts_cbam_results.csv", index=False)
    print("Successfully exported ccts_cbam_results.csv!")
except NameError:
    # If the dataframe variable has a different name, export the active locals
    for var_name in ['df_summary', 'summary_df', 'df_results']:
        if var_name in locals():
            locals()[var_name].to_csv("ccts_cbam_results.csv", index=False)
            print(f"Successfully exported {var_name} to ccts_cbam_results.csv!")
            break