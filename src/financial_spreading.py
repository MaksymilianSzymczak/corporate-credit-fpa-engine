import numpy as np


class CreditSpreadingEngine:

  def __init__(
      self,
      ebitda: float,
      revenue: float,
      total_debt: float,
      cash: float,
      interest_expense: float,
      cfo: float,
      capex: float,
      total_assets: float,
      total_liabilities: float,
      retained_earnings: float,
      working_capital: float,
      ebit: float,
  ):
    self.ebitda = ebitda
    self.revenue = revenue
    self.total_debt = total_debt
    self.cash = cash
    self.net_debt = max(0.0, total_debt - cash)
    self.interest_expense = interest_expense
    self.cfo = cfo
    self.capex = capex
    self.fcf = cfo - capex
    self.total_assets = total_assets
    self.total_liabilities = total_liabilities
    self.retained_earnings = retained_earnings
    self.working_capital = working_capital
    self.ebit = ebit

  def calculate_metrics(self) -> dict:
    gross_leverage = (
        self.total_debt / self.ebitda if self.ebitda > 0 else np.nan
    )
    net_leverage = self.net_debt / self.ebitda if self.ebitda > 0 else np.nan
    interest_coverage = (
        self.ebitda / self.interest_expense
        if self.interest_expense > 0
        else np.nan
    )
    fcf_conversion = self.fcf / self.ebitda if self.ebitda > 0 else np.nan

    # Altman Z-Score dla przedsiębiorstw
    x1 = self.working_capital / self.total_assets if self.total_assets > 0 else 0
    x2 = (
        self.retained_earnings / self.total_assets
        if self.total_assets > 0
        else 0
    )
    x3 = self.ebit / self.total_assets if self.total_assets > 0 else 0
    equity_est = max(0.0, self.total_assets - self.total_liabilities)
    x4 = equity_est / self.total_liabilities if self.total_liabilities > 0 else 0
    x5 = self.revenue / self.total_assets if self.total_assets > 0 else 0
    altman_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5

    return {
        'Net Debt / EBITDA': round(net_leverage, 2),
        'Total Debt / EBITDA': round(gross_leverage, 2),
        'Interest Coverage (ICR)': round(interest_coverage, 2),
        'FCF Conversion': f'{round(fcf_conversion * 100, 1)}%',
        'Altman Z-Score': round(altman_z, 2),
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