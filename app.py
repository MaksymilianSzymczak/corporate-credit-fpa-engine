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

st.title('Corporate Credit Risk, Debt Capacity & P&L Variance Engine')
st.caption('Quantitative Financial & Credit Analytics Platform')

st.sidebar.header('1. Income Statement & Cash Flow (EUR M)')
sales = st.sidebar.number_input('Sales / Revenue (Actual)', value=500.0)
ebitda = st.sidebar.number_input('EBITDA (Actual)', value=85.0)
ebit = st.sidebar.number_input('EBIT (Operating Profit)', value=60.0)
interest_expense = st.sidebar.number_input(
    'Interest Expense (Actual)', value=14.0
)
cfo = st.sidebar.number_input('Operating Cash Flow (CFO)', value=65.0)
capex = st.sidebar.number_input('CapEx', value=25.0)

st.sidebar.header('2. Balance Sheet & Market Data (EUR M)')
total_assets = st.sidebar.number_input('Total Assets', value=600.0)
current_assets = st.sidebar.number_input('Current Assets', value=180.0)
inventory = st.sidebar.number_input('Inventory', value=45.0)
total_liabilities = st.sidebar.number_input('Total Liabilities', value=320.0)
current_liabilities = st.sidebar.number_input(
    'Current Liabilities', value=105.0
)
total_debt = st.sidebar.number_input('Total Debt', value=210.0)
cash = st.sidebar.number_input('Cash & Equivalents', value=30.0)
retained_earnings = st.sidebar.number_input('Retained Earnings', value=140.0)
market_value_equity = st.sidebar.number_input(
    'Market Value of Equity', value=350.0
)

engine = CreditSpreadingEngine(
    sales=sales,
    ebitda=ebitda,
    ebit=ebit,
    interest_expense=interest_expense,
    total_debt=total_debt,
    cash=cash,
    cfo=cfo,
    capex=capex,
    total_assets=total_assets,
    current_assets=current_assets,
    inventory=inventory,
    total_liabilities=total_liabilities,
    current_liabilities=current_liabilities,
    retained_earnings=retained_earnings,
    market_value_equity=market_value_equity,
)

metrics = engine.calculate_metrics()
raw_features = engine.calculate_raw_features()
capacity = engine.calculate_debt_capacity(covenant_max_net_debt_ebitda=3.5)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Net Debt / EBITDA', f"{metrics['Net Debt / EBITDA']}x")
col2.metric(
    'Interest Coverage (ICR)', f"{metrics['Interest Coverage Ratio (ICR)']}x"
)
col3.metric('FCF Conversion Rate', metrics['FCF Conversion Rate'])
col4.metric('Altman Z-Score', metrics['Altman Z-Score'])
col5.metric('Quick Ratio', f"{metrics['Quick Ratio']}x")

st.markdown('---')

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '1. Credit & Debt Capacity',
    '2. P&L Variance & FP&A',
    '3. PD Model & SHAP',
    '4. Macro Stress-Testing',
    '5. Methodology & Formulas',
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
  col_m1, col_m2 = st.columns(2)
  with col_m1:
    st.markdown('**Calculated Financial Metrics**')
    st.json(metrics)
  with col_m2:
    st.markdown('**Altman Z-Score Components ($X_1 - X_5$)**')
    st.json({
        'X1 (Working Capital / Total Assets)': round(
            raw_features['X1_Working_Capital_to_Assets'], 4
        ),
        'X2 (Retained Earnings / Total Assets)': round(
            raw_features['X2_Retained_Earnings_to_Assets'], 4
        ),
        'X3 (EBIT / Total Assets)': round(
            raw_features['X3_EBIT_to_Assets'], 4
        ),
        'X4 (Market Value of Equity / Total Liabilities)': round(
            raw_features['X4_Market_Equity_to_Liabilities'], 4
        ),
        'X5 (Sales / Total Assets)': round(
            raw_features['X5_Sales_to_Assets'], 4
        ),
    })

with tab2:
  st.subheader('FP&A P&L Variance Analysis (Actual vs. Budget)')
  b_col1, b_col2, b_col3, b_col4 = st.columns(4)
  bud_rev = b_col1.number_input('Budget Sales / Revenue', value=480.0)
  bud_ebitda = b_col2.number_input('Budget EBITDA', value=80.0)
  bud_ebit = b_col3.number_input('Budget EBIT', value=56.0)
  bud_int = b_col4.number_input('Budget Interest', value=12.0)

  var_engine = PLVarianceEngine(
      sales,
      ebitda,
      ebit,
      interest_expense,
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
  features = engine.get_model_features()
  pd_result = pd_model.predict_pd(features)

  r_col1, r_col2 = st.columns(2)
  r_col1.metric('Estimated 1-Year PD', pd_result['Probability of Default (PD)'])
  r_col2.metric('Synthetic Credit Rating', pd_result['Synthetic Rating'])

  st.markdown('**Model Input Features (Exact Theoretical Values):**')
  st.dataframe(pd.DataFrame([features]), use_container_width=True)

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
  rev_shock = col_s1.slider('Sales / Revenue Shock (%)', -40, 0, -15) / 100.0
  rate_shock = col_s2.slider('Interest Rate Shock (bps)', 0, 500, 200)

  stress_sim = MacroStressSimulator(
      base_ebitda=ebitda,
      base_ebit=ebit,
      base_debt=engine.net_debt,
      base_interest=interest_expense,
  )
  stress_res = stress_sim.run_scenario(
      revenue_shock_pct=rev_shock, interest_rate_shock_bps=rate_shock
  )
  st.json(stress_res)

with tab5:
  st.subheader('Economic Theory & Mathematical Specification of Features')
  st.markdown(
      'Wszystkie wskaźniki finansowe oraz zmienne wejściowe (features) modelu '
      'ryzyka kredytowego są wyprowadzane bezpośrednio z formalnych definicji '
      'ekonomicznych.'
  )

  st.markdown('### 1. Altman Z-Score Model')
  st.latex(r'\text{Z-Score} = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5')
  st.markdown("""
  Gdzie poszczególne komponenty oznaczają:
  * **$X_1 = \\frac{\\text{Working Capital}}{\\text{Total Assets}}$** — mierzy płynność krótkoterminową, przy czym $\\text{Working Capital} = \\text{Current Assets} - \\text{Current Liabilities}$.
  * **$X_2 = \\frac{\\text{Retained Earnings}}{\\text{Total Assets}}$** — mierzy skumulowaną rentowność w czasie (zdolność do reinwestowania zysków).
  * **$X_3 = \\frac{\\text{EBIT}}{\\text{Total Assets}}$** — mierzy produktywność aktywów przed uwzględnieniem kosztów finansowania i podatków.
  * **$X_4 = \\frac{\\text{Market Value of Equity}}{\\text{Total Liabilities}}$** — mierzy dźwignię rynkową (o ile może spaść wartość aktywów firmy, zanim zobowiązania przekroczą wartość aktywów).
  * **$X_5 = \\frac{\\text{Sales}}{\\text{Total Assets}}$** — mierzy rotację aktywów oraz efektywność wykorzystania kapitału.
  """)

  st.markdown('### 2. Free Cash Flow (FCF) Conversion Rate')
  st.latex(
      r'\text{FCF Conversion Rate} = \left( \frac{\text{Free Cash Flow}}{\text{EBITDA}} \right) \times 100'
  )
  st.latex(
      r'\text{Free Cash Flow (FCF)} = \text{Operating Cash Flow (CFO)} - \text{CapEx}'
  )
  st.markdown(
      'Mierzy zdolność przedsiębiorstwa do zamiany zysku operacyjnego EBITDA '
      'na realną gotówkę dostępną do obsługi zadłużenia.'
  )

  st.markdown('### 3. Interest Coverage Ratio (ICR)')
  st.latex(
      r'\text{Interest Coverage Ratio} = \frac{\text{EBIT}}{\text{Interest Expense}}'
  )
  st.markdown(
      'Określa, ile razy zysk operacyjny (**EBIT** – *Earnings Before Interest '
      'and Taxes*) pokrywa bieżące koszty odsetkowe obsługi długu.'
  )

  st.markdown('### 4. Leverage & Liquidity Ratios (Pozostałe zmienne modelu PD)')
  st.latex(
      r'\text{Net Debt to EBITDA} = \frac{\text{Total Debt} - \text{Cash \& Equivalents}}{\text{EBITDA}}'
  )
  st.latex(
      r'\text{Quick Ratio} = \frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}}'
  )
