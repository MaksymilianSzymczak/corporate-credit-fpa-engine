import pandas as pd


class PLVarianceEngine:

  def __init__(
      self,
      actual_rev: float,
      actual_ebitda: float,
      actual_ebit: float,
      actual_interest: float,
      budget_rev: float,
      budget_ebitda: float,
      budget_ebit: float,
      budget_interest: float,
      tax_rate: float = 0.19,
  ):
    self.actual_rev = actual_rev
    self.actual_ebitda = actual_ebitda
    self.actual_ebit = actual_ebit
    self.actual_interest = actual_interest
    self.budget_rev = budget_rev
    self.budget_ebitda = budget_ebitda
    self.budget_ebit = budget_ebit
    self.budget_interest = budget_interest
    self.tax_rate = tax_rate

  def generate_variance_table(self) -> pd.DataFrame:
    actual_opex = self.actual_rev - self.actual_ebitda
    budget_opex = self.budget_rev - self.budget_ebitda

    actual_ebt = self.actual_ebit - self.actual_interest
    budget_ebt = self.budget_ebit - self.budget_interest

    actual_net = (
        actual_ebt * (1 - self.tax_rate) if actual_ebt > 0 else actual_ebt
    )
    budget_net = (
        budget_ebt * (1 - self.tax_rate) if budget_ebt > 0 else budget_ebt
    )

    rows = [
        ('Revenue', self.budget_rev, self.actual_rev),
        ('Operating Costs (COGS & SG&A)', budget_opex, actual_opex),
        ('EBITDA', self.budget_ebitda, self.actual_ebitda),
        ('EBIT (Operating Profit)', self.budget_ebit, self.actual_ebit),
        ('Interest Expense', self.budget_interest, self.actual_interest),
        ('Net Income', budget_net, actual_net),
    ]

    data = []
    for line, bud, act in rows:
      diff = act - bud
      pct = (diff / bud * 100.0) if bud != 0 else 0.0
      data.append({
          'P&L Line Item': line,
          'Budget (EUR M)': round(bud, 2),
          'Actual (EUR M)': round(act, 2),
          'Variance (EUR M)': round(diff, 2),
          'Variance (%)': f'{round(pct, 1)}%',
      })

    return pd.DataFrame(data)