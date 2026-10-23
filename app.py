import pandas as pd
import plotly.express as px
from src.financial_spreading import CreditSpreadingEngine
from src.pd_model import CreditRiskPDModel
from src.stress_testing import MacroStressSimulator
from src.variance_fpa import PLVarianceEngine
import streamlit as st

st.set_page_config(
    page_title='Corporate Credit Risk & FP&A Variance Engine', layout='wide'
)

st.title('Corporate Credit Risk , Debt Capacity & P&L Variance Engine')
st.caption(
    'Analytics Platform'
)

st.sidebar.header('Actual Financials (EUR M)')
revenue = st.sidebar.number_input('Revenue (Actual)', value=500.0)
ebitda = st.sidebar.number_input('EBITDA (Actual)', value=85.0)
ebit = st.sidebar.number_input('EBIT (Actual)', value=60.0)
total_debt = st.sidebar.number_input('Total Debt', value=210.0)
cash = st.sidebar.number_input('Cash & Equivalents', value=30.0)
interest = st.sidebar.number_input('Interest Expense (Actual)', value=14.0)
cfo = st.sidebar.number_input('Operating Cash Flow (CFO)', value=65.0)
capex = st.sidebar.number_input('CapEx', value=25.0)
total_assets = st.sidebar.number_input('Total Assets', value=600.0)
current_liabilities = st.sidebar.number_input('Current Liabilities', value=320.0)
working_capital = st.sidebar.number_input('Working Capital', value=75.0)
retained_earnings = st.sidebar.number_input('Retained Earnings', value=140.0)

engine = CreditSpreadingEngine(
    ebitda,
    revenue,
    total_debt,
    cash,
    interest,
    cfo,
    capex,
    total_assets,
    current_liabilities,
    retained_earnings,
    working_capital,
    ebit,
)
metrics = engine.calculate_metrics()
capacity = engine.calculate_debt_capacity(covenant_max_net_debt_ebitda=3.5)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Net Debt / EBITDA', f"{metrics['Net Debt / EBITDA']}x")
col2.metric('Interest Coverage (ICR)', f"{metrics['Interest Coverage (ICR)']}x")
col3.metric('Altman Z-Score', metrics['Altman Z-Score'])
col4.metric(
    'Debt Headroom', f"{capacity['Additional Debt Capacity (Headroom)']} M EUR"
)

st.markdown('---')

tab1, tab2, tab3, tab4 = st.tabs([
    '1. Credit & Debt Capacity',
    '2. P&L Variance & FP&A',
    '3. PD Model & SHAP',
    '4. Macro Stress-Testing',
])

with tab1:
  st.subheader('Covenant Compliance & Debt Sizing')
  if capacity['Covenant Breach']:
    st.error('COVENANT BREACH: Net Debt / EBITDA exceeds 3.5x limit!')
  else:
    st.success(
        'Covenant Compliant. Available Debt Headroom: '
        f"**{capacity['Additional Debt Capacity (Headroom)']} M EUR**."
    )
  st.json(metrics)

with tab2:
  st.subheader('FP&A P&L Variance Analysis (Actual vs. Budget)')
  b_col1, b_col2, b_col3, b_col4 = st.columns(4)
  bud_rev = b_col1.number_input('Budget Revenue', value=480.0)
  bud_ebitda = b_col2.number_input('Budget EBITDA', value=80.0)
  bud_ebit = b_col3.number_input('Budget EBIT', value=56.0)
  bud_int = b_col4.number_input('Budget Interest', value=12.0)

  var_engine = PLVarianceEngine(
      revenue,
      ebitda,
      ebit,
      interest,
      bud_rev,
      bud_ebitda,
      bud_ebit,
      bud_int,
  )
  var_df = var_engine.generate_variance_table()
  st.dataframe(var_df, use_container_width=True)

with tab3:
  st.subheader('Probability of Default (PD) & Explainable AI (SHAP)')
  pd_model = CreditRiskPDModel()
  fcf_conv = (cfo - capex) / ebitda if ebitda > 0 else 0
  quick_ratio = (
      working_capital / total_liabilities if total_liabilities > 0 else 1.0
  )

  features = {
      'Net_Debt_EBITDA': metrics['Net Debt / EBITDA'],
      'Interest_Coverage': metrics['Interest Coverage (ICR)'],
      'FCF_Conversion': fcf_conv,
      'Altman_Z': metrics['Altman Z-Score'],
      'Quick_Ratio': round(quick_ratio, 2),
  }
  pd_result = pd_model.predict_pd(features)

  r_col1, r_col2 = st.columns(2)
  r_col1.metric('Estimated 1-Year PD', pd_result['Probability of Default (PD)'])
  r_col2.metric('Synthetic Credit Rating', pd_result['Synthetic Rating'])

  shap_df = pd.DataFrame(
      list(pd_result['SHAP_values'].items()),
      columns=['Feature', 'SHAP Risk Contribution'],
  ).sort_values(by='SHAP Risk Contribution', ascending=True)

  fig = px.bar(
      shap_df,
      x='SHAP Risk Contribution',
      y='Feature',
      orientation='h',
      color='SHAP Risk Contribution',
      color_continuous_scale='RdYlGn_r',
  )
  st.plotly_chart(fig, use_container_width=True)

with tab4:
  st.subheader('Macroeconomic Stress-Testing Simulator')
  col_s1, col_s2 = st.columns(2)
  rev_shock = col_s1.slider('Revenue Shock (%)', -40, 0, -15) / 100.0
  rate_shock = col_s2.slider('Interest Rate Shock (bps)', 0, 500, 200)

  stress_sim = MacroStressSimulator(
      base_ebitda=ebitda, base_debt=total_debt - cash, base_interest=interest
  )
  stress_res = stress_sim.run_scenario(
      revenue_shock_pct=rev_shock, interest_rate_shock_bps=rate_shock
  )
  st.json(stress_res)
