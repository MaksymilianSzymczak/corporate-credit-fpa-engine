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
    self.total_liabilities = total_liabilities
    self.current_liabilities = current_liabilities
    self.retained_earnings = retained_earnings
    self.market_value_equity = market_value_equity

    # Wyprowadzone wielkości fundamentalne
    self.net_debt = max(0.0, self.total_debt - self.cash)
    self.free_cash_flow = self.cfo - self.capex
    self.working_capital = self.current_assets - self.current_liabilities

  def calculate_raw_features(self) -> dict:
    """Oblicza dokładne wartości liczbowe wskaźników zgodnie z teorią ekonomiczną."""
    # 1. Dźwignia finansowa
    gross_leverage = (
        self.total_debt / self.ebitda if self.ebitda > 0 else np.nan
    )
    net_debt_to_ebitda = (
        self.net_debt / self.ebitda if self.ebitda > 0 else np.nan
    )

    # 2. Interest Coverage Ratio = EBIT / Interest Expense
    interest_coverage_ratio = (
        self.ebit / self.interest_expense
        if self.interest_expense > 0
        else np.nan
    )

    # 3. FCF Conversion Rate = (Free Cash Flow / EBITDA) * 100
    fcf_conversion_rate = (
        (self.free_cash_flow / self.ebitda) * 100.0
        if self.ebitda > 0
        else np.nan
    )

    # 4. Altman Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
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

    altman_z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # 5. Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    quick_ratio = (
        (self.current_assets - self.inventory) / self.current_liabilities
        if self.current_liabilities > 0
        else np.nan
    )

    return {
        'Total_Debt_to_EBITDA': gross_leverage,
        'Net_Debt_to_EBITDA': net_debt_to_ebitda,
        'Interest_Coverage_Ratio': interest_coverage_ratio,
        'FCF_Conversion_Rate': fcf_conversion_rate,
        'Altman_Z_Score': altman_z_score,
        'Quick_Ratio': quick_ratio,
        'X1_Working_Capital_to_Assets': x1,
        'X2_Retained_Earnings_to_Assets': x2,
        'X3_EBIT_to_Assets': x3,
        'X4_Market_Equity_to_Liabilities': x4,
        'X5_Sales_to_Assets': x5,
    }

  def calculate_metrics(self) -> dict:
    """Zwraca sformatowane wskaźniki do prezentacji w interfejsie."""
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
    """Zwraca słownik zmiennych (features) gotowy do podania do modelu PD."""
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
