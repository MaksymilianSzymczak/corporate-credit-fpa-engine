import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


class CreditRiskPDModel:

  def __init__(self):
    self.model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    self.feature_names = [
        'Net_Debt_EBITDA',
        'Interest_Coverage',
        'FCF_Conversion',
        'Altman_Z',
        'Quick_Ratio',
    ]
    self._is_trained = False
    self.explainer = None

  def generate_synthetic_training_data(self, n_samples: int = 1500):
    np.random.seed(42)
    net_debt_ebitda = np.random.normal(loc=2.8, scale=1.5, size=n_samples)
    interest_coverage = np.random.normal(loc=4.5, scale=2.5, size=n_samples)
    fcf_conversion = np.random.normal(loc=0.35, scale=0.25, size=n_samples)
    altman_z = np.random.normal(loc=2.5, scale=1.0, size=n_samples)
    quick_ratio = np.random.normal(loc=1.2, scale=0.5, size=n_samples)

    log_odds = (
        -4.2
        + 0.85 * net_debt_ebitda
        - 0.45 * interest_coverage
        - 1.5 * fcf_conversion
        - 0.9 * altman_z
        - 0.8 * quick_ratio
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, prob_default)

    return pd.DataFrame({
        'Net_Debt_EBITDA': net_debt_ebitda,
        'Interest_Coverage': interest_coverage,
        'FCF_Conversion': fcf_conversion,
        'Altman_Z': altman_z,
        'Quick_Ratio': quick_ratio,
        'Default': y,
    })

  def train_model(self):
    df = self.generate_synthetic_training_data()
    X = df[self.feature_names]
    y = df['Default']
    self.model.fit(X, y)
    self._is_trained = True
    self.explainer = shap.Explainer(self.model, X)

  def predict_pd(self, features_dict: dict) -> dict:
    if not self._is_trained:
      self.train_model()

    df_sample = pd.DataFrame([features_dict])[self.feature_names]
    pd_value = float(self.model.predict_proba(df_sample)[0][1])

    if pd_value < 0.001:
      rating = 'AAA'
    elif pd_value < 0.003:
      rating = 'AA'
    elif pd_value < 0.008:
      rating = 'A'
    elif pd_value < 0.020:
      rating = 'BBB'
    elif pd_value < 0.050:
      rating = 'BB'
    elif pd_value < 0.120:
      rating = 'B'
    else:
      rating = 'CCC/D'

    shap_values = self.explainer(df_sample)

    return {
        'Probability of Default (PD)': f'{round(pd_value * 100, 2)}%',
        'PD_numeric': pd_value,
        'Synthetic Rating': rating,
        'SHAP_values': dict(zip(self.feature_names, shap_values.values[0])),
    }