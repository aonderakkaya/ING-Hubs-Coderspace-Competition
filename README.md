# ING Hubs Türkiye Datathon 2025: Customer Churn Prediction

This repository contains the end-to-end machine learning model I developed for the 2025 Datathon, organized by ING Hubs Türkiye. This marks my **first-ever Kaggle competition** and my **first AI project built from scratch**.

## 🚀 About the Datathon

This first Datathon, organized by ING Hubs Türkiye (founded in 2024), aimed to bring together AI enthusiasts and engineers to create data-driven solutions for real-world banking problems.

### 🎯 Objective of the Competition

The primary goal of the competition was to develop a classification model to predict the **probability of customer churn** within the six-month period following a given reference date.

---

## 🛠️ My Technical Approach and Solution

A hybrid approach was used in this project, combining high-performance data processing, advanced feature engineering, and industry-standard modeling techniques.

### 1. Data Processing: Polars & Pandas

* **High-Performance Data Loading:** The **Polars** library was used instead of standard Pandas to efficiently process large datasets. Date columns (`date`, `ref_date`) were converted to the correct data type directly upon reading (`pl.read_csv`).
* **Hybrid Model:** All heavy data processing and feature engineering were performed using Polars' **LazyFrame** API. For the modeling phase, the data was strategically converted to a Pandas DataFrame using `.to_pandas()` to ensure compatibility with Scikit-learn.

### 2. Advanced Feature Engineering

The most critical part of this project was deriving temporal features from customer behavior.

* **Data Leakage Prevention:** When joining the customer history (`customer_history`), a **`pl.col("date") < pl.col("ref_date")`** filter was applied to ensure temporal integrity and prevent the model from seeing future information.
* **Historical Aggregations:** Statistical features such as `mean`, `sum`, `max`, and `count` were generated for critical variables like `mobile_eft_all_cnt` and `cc_transaction_all_amt`.
* **Recency Features:**
    * A `days_since_last_transaction` feature was created by calculating the difference in days between the reference date and the customer's last transaction.
    * `_last_month` features (e.g., `cc_transaction_all_amt_last_month`) were derived to capture the customer's most recent behavior.
* **Trend Features:** Features like `cc_spend_trend_ratio` (last month's spend / average spend) were created to capture changes in customer spending habits.

### 3. Pre-processing and Pipeline

* **Class Imbalance:** The `scale_pos_weight` was calculated as **6.06** to handle the minority "churn" class. In the final model, this value was **aggressively increased by a factor of 1.3** to forcefully improve the Recall/Lift metrics.
* **Scikit-learn Pipeline:** A clean preprocessing pipeline was established using `ColumnTransformer`.
* **Advanced Encoding:** When using `OneHotEncoder` for categorical variables, the **`min_frequency=500`** parameter was set. This is an advanced technique to prevent noise and overfitting by only encoding categories that appear at least 500 times.

### 4. Modeling: XGBoost

* **`xgboost.XGBClassifier`** was chosen as the model, as it is the industry standard for such tabular data and imbalance problems.
* The model was trained on the entire training dataset using optimized hyperparameters (low `learning_rate`, high `n_estimators`, tuned `max_depth`, and the aggressive `scale_pos_weight`) to generate the final predictions.

---

## 📊 Evaluation Metric and Results

Success was measured by a custom score, which was a weighted sum of three metrics: Gini, Recall, and Lift. This score showed the model's performance relative to a provided baseline.

* **Gini:** 40% (Related to ROC AUC: `Gini = 2 * ROC AUC - 1`)
* **Recall@10%:** 30%
* **Lift@10%:** 30%

**Baseline Model Metrics:**
* **Gini:** 0.38515
* **Recall@10%:** 0.18469
* **Lift@10%:** 1.84715

---

## 📈 My Results and Retrospective

* **My Score:** 1.05
* **Total Submissions:** 3

This competition was a fantastic opportunity for me, as a statistics student, to apply my theoretical knowledge to a practical problem.

While the winning scores were around 1.20 (achieved with 30-40+ submissions), **I am proud to have achieved a score of 1.05 with only 3 submissions**. This demonstrates my ability to quickly understand a problem and implement an effective baseline model.

This project proved to me that I can work on an AI problem from scratch using tools like Python and Scikit-learn and participate in a complex data science competition.

### Future Work

If I were to continue this project, my next steps would focus on:
* Conducting more in-depth **Feature Engineering**.
* Experimenting with different model algorithms (e.g., LightGBM, CatBoost).
* Applying **Hyperparameter Optimization** (e.g., GridSearchCV, Optuna).
