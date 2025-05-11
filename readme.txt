🧠 automl_comparison.py
Purpose:
Benchmark and compare popular AutoML libraries (e.g. H2O, TPOT, PyCaret, Auto-Sklearn) for both classification and regression tasks.

Key Features:

💡 Automatically selects best models with minimal manual tuning

📊 Compares performance metrics (e.g. accuracy, R², AUC) across libraries

🧪 Cross-validation support for robust evaluation

🔍 Visual output of leaderboard results

When to Use:

Need a strong baseline fast

Want to compare AutoML tools on your dataset

For POCs or to accelerate model selection

🧪 classification_models.py
Purpose:
Train, evaluate, and compare supervised classification models using Scikit-learn and optionally XGBoost/LightGBM.

Models Included:

Logistic Regression

Random Forest

Decision Tree

SVM

KNN

Gradient Boosting (XGBoost/LightGBM)

Key Features:

🔁 Cross-validation + confusion matrix, ROC-AUC, precision, recall

🔍 Grid search / hyperparameter tuning support

📊 Model comparison table

When to Use:

Binary or multi-class classification tasks

Benchmarking ML classifiers

Model explainability via feature importance

🛠️ feature_engineering.py
Purpose:
Perform structured feature transformations to improve predictive power before model training.

Included Techniques:

Missing value imputation strategies

One-hot and label encoding

Binning, scaling, log transforms

Date/time feature expansion

Feature selection (correlation + variance-based)

Key Features:

✅ Modular pipeline structure

🧪 Easy integration into ML pipelines

⚙️ Works with both classification and regression targets

When to Use:

Before any ML model training

During preprocessing pipeline creation

To improve model accuracy & reduce noise

📈 forecasting_arima_prophet.py
Purpose:
Time series forecasting using ARIMA and Meta’s Prophet.

Key Features:

📅 Works with time-indexed Pandas DataFrames

🔁 Rolling forecast / backtesting support

🧠 Automatically finds optimal parameters (Prophet/ARIMA)

📊 Plot forecast + confidence intervals

When to Use:

Forecasting KPIs, sales, trends, etc.

Comparing classical (ARIMA) vs. modern (Prophet) approaches

Generating future predictions with seasonality & holidays

📉 regression_models.py
Purpose:
Train and compare supervised regression models on numerical targets.

Models Included:

Linear Regression

Lasso, Ridge, ElasticNet

Random Forest Regressor

Gradient Boosting Regressor

XGBoost/LightGBM (optional)

Key Features:

🔁 Cross-validation with RMSE, MAE, R²

📊 Residual plots + feature importance

⚙️ Grid search tuning and pipeline compatibility

When to Use:

House pricing, demand prediction, or any regression problem

Need interpretable vs. high-performance models

For evaluation of ensemble vs. linear models
