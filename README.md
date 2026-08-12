# Volunteer Project: EDA and ML Classification of Donors for the "Bureau of Good Deeds" Foundation

Data Science & Machine Learning project focusing on donor behavior analysis for a charitable foundation to optimize fundraising campaigns and predict churn.

## Objective
Identify donors from the active pool of the past year who are potentially ready to increase their contribution frequency or donation size. Additionally, detect behavior markers of users who are prone to stopping their donations.

## Data
* Real historical data from the foundation covering approximately 4,000 donors over several years (transactions, payment dynamics, email activity).

## Tech Stack
* Language: Python
* ML Models: CatBoost, Random Forest
* Metrics: RFM Analysis (Recency, Frequency, Monetary)
* Libraries: Pandas, NumPy, Matplotlib, Seaborn
* Tools: Jupyter Notebook

## Key Implementations
1. Data Preprocessing and Cleaning: Handled missing values and removed duplicates. Categorized locations (Moscow, St. Petersburg, other Russian regions, abroad) and payment purposes (general donation, named donation, SMS).
2. Feature Engineering: Built an aggregated dataset per donor (1,206 rows). Engineered features including lifetime, time elapsed since the first/last payment and email activity, total donation amount, median and average check, and average period between donations.
3. Donor Segmentation: Categorized donors into 4 classes utilizing the RFM methodology.
4. Model Training: Prepared and trained baseline classification models without hyperparameter tuning:
   * CatBoost Classifier: Accuracy = 0.90
   * Random Forest: Accuracy = 0.87

## Future Roadmap
* Expand Feature Engineering: Add features tracking changes in the average donation amount (first half vs. second half of the donor's lifetime).
* Hyperparameter Tuning: Implement GridSearchCV / Optuna to enhance model performance.
* Cohort Analysis: Isolate new users (up to 6 months of activity) into a separate dataset to forecast their future RFM class.
