# Corporate Credit Risk, Debt Capacity & FP&A Variance Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ms-corporate-credit-fpa-engine.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/MaksymilianSzymczak/corporate-credit-fpa-engine)

> 🔗 **Interactive Live Demo:** [ms-corporate-credit-fpa-engine.streamlit.app](https://ms-corporate-credit-fpa-engine.streamlit.app/)

An end-to-end quantitative financial analytics platform combining **Corporate Credit Analysis (GCB)**, **Financial Planning & P&L Variance Analytics (GF&BM)**, and **Machine Learning Credit Risk Modeling (Explainable AI / Basel IRB framework)**.

---

## Key Modules & Features

### 1. Corporate Credit Spreading & Debt Capacity (GCB)
* **Automated 3-Statement Spreading:** Computes core leverage and coverage metrics strictly aligned with economic theory:
  * $\text{Net Debt / EBITDA} = \frac{\text{Total Debt} - \text{Cash}}{\text{EBITDA}}$
  * $\text{Interest Coverage Ratio (ICR)} = \frac{\text{EBIT}}{\text{Interest Expense}}$
  * $\text{FCF Conversion Rate} = \left(\frac{\text{CFO} - \text{CapEx}}{\text{EBITDA}}\right) \times 100$
  * $\text{Quick Ratio} = \frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}}$
* **Classic Altman Z-Score:** Implements the original 1968 manufacturing formula utilizing the Market Value of Equity:
  * $\text{Z-Score} = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$
* **Covenant Testing & Debt Sizing:** Calculates available borrowing capacity (*Debt Headroom*) against contractual maintenance covenant thresholds (e.g., $\text{Net Debt / EBITDA} \le 3.5\text{x}$).

### 2. FP&A & P&L Variance Analytics (GF&BM / Product Control)
* **Budget vs. Actual Variance Engine:** Reconciles operating performance across Sales/Revenue, OPEX, EBITDA, EBIT, Interest, and Net Income.
* **Variance Decomposition:** Quantifies both absolute ($\Delta \text{ EUR M}$) and percentage ($\Delta\%$) deviations to support executive decision-making and reporting accuracy.

### 3. Credit Risk PD Model & Explainable AI (Data Science & Risk)
* **Probability of Default (PD) Classifier:** Logistic regression model calibrated on synthetic regulatory corporate default data using formal credit features.
* **Synthetic Credit Rating:** Maps estimated 1-year default probability into agency-style rating grades (AAA to CCC/D).
* **Explainable AI (SHAP):** Decomposes individual balance sheet and income statement risk contributions using Shapley additive explanations.

### 4. Macroeconomic Stress-Testing Simulator
* **Scenario Modeling:** Simulates adverse macroeconomic shocks, such as simultaneous revenue contractions and central bank interest rate hikes (+200 bps).
* **Covenant Breach Early Warning:** Evaluates dynamic post-shock leverage ratios and interest coverage resilience (EBIT-driven).

### 5. Methodology & Formulas Tab
* **Interactive Formula Reference:** A dedicated application tab presenting formal LaTeX equations and economic definitions for all underlying metrics, ensuring full transparency of the analytical engine.

---

## Tech Stack
* **Language:** Python 3.10+
* **Data Processing & Analytics:** `pandas`, `numpy`, `scikit-learn`
* **Explainability:** `shap`
* **Visualization & UI:** `streamlit`, `plotly`

---

## Getting Started

```bash

# Clone the repository
git clone https://github.com/MaksymilianSzymczak/corporate-credit-fpa-engine.git

# Install required dependencies
pip install -r requirements.txt

# Run the interactive Streamlit dashboard
streamlit run app.py
