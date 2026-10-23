import numpy as np


class CreditSpreadingEngine:

  def __init__(
      self,
      sales: float,
      ebitda: float,
      ebit: float,
      interest_expense: float,
      total_debt: float,
      cash: float,
      cfo: float,
      capex: float,
      total_assets: float,
      current_assets: float,
      inventory: float,
      prepaid_expenses: float,  # Dodana pozycja bilansowa
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
    self.cash = cash
    self.cfo = cfo
    self.capex = capex
    self.total_assets = total_assets
    self.current_assets = current_assets
    self.inventory = inventory
    self.prepaid_expenses = prepaid_expenses
    self.total_liabilities = total_liabilities
    self.current_liabilities = current_liabilities
    self.retained_earnings = retained_earnings
    self.market_value_equity = market_value_equity

    # Wyprowadzone wielkości fundamentalne
    self.net_debt = max(0.0, self.total_debt - self.cash)
    self.free_cash_flow = self.cfo - self.capex
    self.working_capital = self.current_assets - self.current_liabilities

    # Pełny formalny wzór Quick Ratio
    quick_assets = self.current_assets - self.inventory - self.prepaid_expenses
    self.quick_ratio = (
        quick_assets / self.current_liabilities
        if self.current_liabilities > 0
        else np.nan
    )

  def calculate_quick_ratio(self) -> float:
    return self.quick_ratio

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
        'Working Capital (EUR M)': round(self.working_capital, 2),
        'Free Cash Flow (EUR M)': round(self.free_cash_flow, 2),
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

  def calculate_debt_capacity(
      self, covenant_max_net_debt_ebitda: float = 3.5
  ) -> dict:
    max_allowable_net_debt = self.ebitda * covenant_max_net_debt_ebitda
    additional_debt_headroom = max_allowable_net_debt - self.net_debt
    return {
        'Covenant Threshold': f'{covenant_max_net_debt_ebitda}x',
        'Current Net Debt': round(self.net_debt, 2),
        'Max Net Debt Allowed': round(max_allowable_net_debt, 2),
        'Additional Debt Capacity (Headroom)': round(
            additional_debt_headroom, 2
        ),
        'Covenant Breach': self.net_debt > max_allowable_net_debt,
    }
