import numpy as np


class CreditSpreadingEngine:

  def __init__(
      self,
      sales: float,
      ebitda: float,
      ebit: float,
      interest_expense: float,
      total_debt: float,
      cash_and_equivalents: float,
      accounts_receivable: float,  # Net Accounts Receivable (NAR)
      marketable_securities: float,  # Marketable Securities (MS)
      cfo: float,
      capex: float,
      total_assets: float,
      current_assets: float,
      total_liabilities: float,
      current_liabilities: float,
      retained_earnings: float,
      market_value_equity: float,
  ):
    self.sales = sales
    self.ebitda = ebitda
    self.ebit = ebit
    self.interest_expense = interest_expense
    self.total_debt = total_debt
    self.cash = cash_and_equivalents
    self.accounts_receivable = accounts_receivable
    self.marketable_securities = marketable_securities
    self.cfo = cfo
    self.capex = capex
    self.total_assets = total_assets
    self.current_assets = current_assets
    self.total_liabilities = total_liabilities
    self.current_liabilities = current_liabilities
    self.retained_earnings = retained_earnings
    self.market_value_equity = market_value_equity

    # Pozycje wyprowadzone
    self.net_debt = max(
        0.0, self.total_debt - (self.cash + self.marketable_securities)
    )
    self.free_cash_flow = self.cfo - self.capex
    self.working_capital = self.current_assets - self.current_liabilities

    # ADDYTYWNE QUICK ASSETS: Cash + CE + MS + NAR
    self.quick_assets = (
        self.cash + self.marketable_securities + self.accounts_receivable
    )
    self.quick_ratio = (
        self.quick_assets / self.current_liabilities
        if self.current_liabilities > 0
        else np.nan
    )

  def calculate_raw_features(self) -> dict:
    net_debt_to_ebitda = (
        self.net_debt / self.ebitda if self.ebitda > 0 else np.nan
    )
    gross_leverage = (
        self.total_debt / self.ebitda if self.ebitda > 0 else np.nan
    )
    interest_coverage = (
        self.ebit / self.interest_expense
        if self.interest_expense > 0
        else np.nan
    )
    fcf_conversion = (
        (self.free_cash_flow / self.ebitda) * 100.0
        if self.ebitda > 0
        else np.nan
    )

    x1 = (
        self.working_capital / self.total_assets
        if self.total_assets > 0
        else 0.0
    )
    x2 = (
        self.retained_earnings / self.total_assets
        if self.total_assets > 0
        else 0.0
    )
    x3 = self.ebit / self.total_assets if self.total_assets > 0 else 0.0
    x4 = (
        self.market_value_equity / self.total_liabilities
        if self.total_liabilities > 0
        else 0.0
    )
    x5 = self.sales / self.total_assets if self.total_assets > 0 else 0.0
    altman_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    return {
        'Net_Debt_to_EBITDA': net_debt_to_ebitda,
        'Interest_Coverage_Ratio': interest_coverage,
        'FCF_Conversion_Rate': fcf_conversion,
        'Altman_Z_Score': altman_z,
        'Quick_Ratio': self.quick_ratio,
        'Total_Debt_to_EBITDA': gross_leverage,
        'X1': x1,
        'X2': x2,
        'X3': x3,
        'X4': x4,
        'X5': x5,
    }

  def calculate_metrics(self) -> dict:
    raw = self.calculate_raw_features()
    return {
        'Net Debt / EBITDA': round(raw['Net_Debt_to_EBITDA'], 2),
        'Total Debt / EBITDA': round(raw['Total_Debt_to_EBITDA'], 2),
        'Interest Coverage Ratio (ICR)': round(
            raw['Interest_Coverage_Ratio'], 2
        ),
        'FCF Conversion Rate': f"{round(raw['FCF_Conversion_Rate'], 1)}%",
        'Altman Z-Score': round(raw['Altman_Z_Score'], 2),
        'Quick Ratio': round(raw['Quick_Ratio'], 2),
        'Quick Assets (EUR M)': round(self.quick_assets, 2),
        'Working Capital (EUR M)': round(self.working_capital, 2),
    }

  def get_model_features(self) -> dict:
    raw = self.calculate_raw_features()
    return {
        'Net_Debt_to_EBITDA': round(raw['Net_Debt_to_EBITDA'], 4),
        'Interest_Coverage_Ratio': round(raw['Interest_Coverage_Ratio'], 4),
        'FCF_Conversion_Rate': round(raw['FCF_Conversion_Rate'], 4),
        'Altman_Z_Score': round(raw['Altman_Z_Score'], 4),
        'Quick_Ratio': round(raw['Quick_Ratio'], 4),
    }
