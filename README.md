# 🚀 Feature Shipment Delay Prediction — American Express
**Technical Assessment | AI/ML Engineer Role | Britha AI**

Predicts the number of days a software feature shipment will be delayed,
enabling teams to produce accurate delivery windows before a sprint begins.

---

## 📊 Results

| Metric | Baseline XGBoost | Fine-Tuned XGBoost | Improvement |
|--------|:-:|:-:|:-:|
| **MAE** | 1.025 days | **0.889 days** | ↓ 13.3% |
| **RMSE** | 1.288 days | **1.109 days** | ↓ 13.9% |
| **R²** | 0.9327 | **0.9502** | ↑ 1.9pp |
| **5-Fold CV MAE** | — | **0.897 ± 0.010** | Stable, no overfitting |

> The fine-tuned model predicts shipment delay to within **~0.89 days** on unseen data.

---

## 📁 Project Structure
---

## 🛠️ How to Run

**Option 1 — Google Colab (recommended, no setup needed):**
1. Open the `.ipynb` notebook file above
2. Click "Open in Colab" 
3. Upload the CSV when prompted
4. Run all cells top to bottom

**Option 2 — Local:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib
jupyter notebook feature_shipment_prediction.ipynb
```

---

## 🧠 Methodology

### 1. EDA Findings
- 1,300 rows, 11 columns, **zero missing values**
- `delay_days` ranges 2–29, mean ~13.4 days
- Strongest correlations: `feature_complexity`, `past_avg_delay_days`, `estimated_bug_count`

### 2. Feature Engineering (20 features total)
| Type | Features Added |
|------|---------------|
| Temporal | `month`, `quarter`, `day_of_week`, `week_of_year`, `is_month_end`, `is_quarter_end` |
| Interaction | `complexity_x_bugs`, `blockers_per_dep`, `team_efficiency`, `sprint_load` |
| Risk | `delay_risk_score` (weighted composite of key risk factors) |

### 3. Why XGBoost?
- Handles **non-linear feature interactions** natively
- Built-in **L1/L2 regularization** — robust to noise and outliers
- Outperforms linear models on structured tabular data
- Fast training with interpretable feature importances

### 4. Fine-Tuning: RandomizedSearchCV
80 iterations × 5-fold cross-validation, optimizing for minimum MAE:

| Hyperparameter | Search Range | Best Value |
|---|---|---|
| `n_estimators` | 100–600 | 552 |
| `max_depth` | 3–10 | 3 |
| `learning_rate` | 0.01–0.30 | 0.023 |
| `subsample` | 0.60–1.00 | 0.797 |
| `colsample_bytree` | 0.50–1.00 | 0.507 |
| `reg_alpha` (L1) | 0–1 | 0.474 |
| `reg_lambda` (L2) | 0.5–2.5 | 0.696 |

**Why RandomizedSearch over GridSearch?** With 9 hyperparameters and continuous ranges, RandomizedSearch explores the space ~10× more efficiently at the same compute budget.

---

## 🔑 Top Predictive Features
1. `past_avg_delay_days` — history is the best predictor
2. `feature_complexity` — more complex = more unknowns
3. `complexity_x_bugs` — compound risk signal
4. `delay_risk_score` — engineered weighted composite
5. `estimated_bug_count` — quality debt at shipment time

---

## 🔮 Next Shipment Prediction
Given a feature planned for **2025-07-14**:
- **Predicted delay:** ~17.6 days
- **Expected ship date: 2025-08-01**

---

## 🚀 Future Improvements
1. **Ensemble stacking** — XGBoost + LightGBM + CatBoost
2. **SHAP values** — per-prediction explanations for stakeholders
3. **TimeSeriesSplit CV** — respect temporal ordering in validation
4. **Quantile regression** — output confidence intervals (e.g. "80% chance ships by X")
5. **More signals** — PR merge velocity, code review time, team tenure
