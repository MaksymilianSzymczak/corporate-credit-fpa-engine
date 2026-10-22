class MacroStressSimulator:

  def __init__(self, base_ebitda: float, base_debt: float, base_interest: float):
    self.base_ebitda = base_ebitda
    self.base_debt = base_debt
    self.base_interest = base_interest

  def run_scenario(
      self,
      revenue_shock_pct: float = -0.15,
      interest_rate_shock_bps: float = 200,
      covenant_threshold: float = 3.5,
  ) -> dict:
    operating_leverage = 1.8
    stressed_ebitda = max(
        1.0,
        self.base_ebitda * (1 + revenue_shock_pct * operating_leverage),
    )
    additional_interest = self.base_debt * (interest_rate_shock_bps / 10000.0)
    stressed_interest = self.base_interest + additional_interest

    stressed_leverage = self.base_debt / stressed_ebitda
    stressed_icr = stressed_ebitda / stressed_interest

    return {
        'Stressed EBITDA': round(stressed_ebitda, 2),
        'Stressed Interest Expense': round(stressed_interest, 2),
        'Stressed Net Debt / EBITDA': round(stressed_leverage, 2),
        'Stressed Interest Coverage': round(stressed_icr, 2),
        'Covenant Breach Risk': stressed_leverage > covenant_threshold,
    }