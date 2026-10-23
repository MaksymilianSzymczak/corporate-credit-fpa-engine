class MacroStressSimulator:

  def __init__(
      self,
      base_ebitda: float,
      base_ebit: float,
      base_debt: float,
      base_interest: float,
  ):
    self.base_ebitda = base_ebitda
    self.base_ebit = base_ebit
    self.base_debt = base_debt
    self.base_interest = base_interest

  def run_scenario(
      self,
      revenue_shock_pct: float = -0.15,
      interest_rate_shock_bps: float = 200,
      covenant_threshold: float = 3.5,
  ) -> dict:
    operating_leverage = 1.8
    shock_factor = 1.0 + (revenue_shock_pct * operating_leverage)

    stressed_ebitda = max(1.0, self.base_ebitda * shock_factor)
    stressed_ebit = max(0.5, self.base_ebit * shock_factor)

    additional_interest = self.base_debt * (interest_rate_shock_bps / 10000.0)
    stressed_interest = self.base_interest + additional_interest

    stressed_leverage = self.base_debt / stressed_ebitda
    # Poprawiony wzór ICR w stress-teście: EBIT / Interest Expense
    stressed_icr = (
        stressed_ebit / stressed_interest if stressed_interest > 0 else 0.0
    )

    return {
        'Stressed EBITDA': round(stressed_ebitda, 2),
        'Stressed EBIT': round(stressed_ebit, 2),
        'Stressed Interest Expense': round(stressed_interest, 2),
        'Stressed Net Debt / EBITDA': round(stressed_leverage, 2),
        'Stressed Interest Coverage Ratio (EBIT / Interest)': round(
            stressed_icr, 2
        ),
        'Covenant Breach Risk': stressed_leverage > covenant_threshold,
    }
