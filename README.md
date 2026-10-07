Bank Customer Churn Analysis

An end-to-end churn analysis for a bank, built as a business-analyst exercise: from raw data, to findings and recommendations, to a Streamlit dashboard and a baseline prediction model.

Business question: Who is leaving the bank, and where should retention effort go?

Dataset
Source: Kaggle, Bank Customer Churn dataset (Churn_Modelling.csv)
Size: 10,000 customers, 14 columns. No duplicates and no missing values.
Target: Exited (1 = customer left the bank, 0 = stayed)
Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
Identifiers (not used for analysis): RowNumber, CustomerId, Surname
Key findings

Overall churn is 20.4% (2,037 of 10,000 customers). All rates below are measured inside each group.

Factor	Finding
Age	Customers aged 49-60 churn at 55.1%, about 2.7x the bank average. They are 29.2% of all leavers.
Country	Germany churns at 32.4%, versus 16.2% (France) and 16.7% (Spain). High for both genders.
Gender	Women churn at 25.1%, men at 16.5%. The gap appears in all three countries and age does not explain it.
Products	1 product: 27.7% churn (50.8% of customers, 69.2% of leavers). 2 products: only 7.6%. 3 products: 82.7% (n=266). 4 products: 100% (n=60).
Activity	Inactive members churn at 26.85%, active members at 14.27% (about 1.9x).
Balance	100k-150k: 25.8% and 150k+: 23.0%, above average. Zero balance: 13.8% (lowest).
Recommendations
Run proactive retention for customers aged 49-60, starting in Germany.
Survey leavers in Germany and women leavers to find out why they leave.
Test cross-selling a 2nd product to 1-product customers with an A/B test.
Use falling activity as an early-warning signal.
Give high-balance customers (100k+) priority retention attention.
Model

A baseline churn predictor to shortlist customers for retention calls.

Algorithm: Random Forest (400 trees, max depth 7)
Features: one-hot Geography and Gender, plus Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember
Preprocessing: MinMaxScaler, 67/33 train-test split (random_state=42)
Results (test set): accuracy about 0.87, churn recall about 0.4, churn precision about 0.8
Baseline: predicting "everyone stays" already gives about 0.80 accuracy, so recall on churners is the metric that matters.

The model catches roughly 4 of every 10 customers who leave. Adding Age, Geography and Gender improved churn recall from about 0.15 to about 0.4.

Limitations
The data shows who leaves, not why. Reasons need surveys or interviews.
Activity and number of products are associations, not proven causes. Confirm with an A/B test.
Groups with few customers (3-4 products, the 1-50k balance group) give unreliable rates.
The Random Forest has no fixed random_state, so metrics vary slightly between runs.
Recall is still low: the model misses many customers who leave.
Project structure
├── banking.ipynb            # analysis and model (notebook)
├── banking_streamlit.py     # interactive dashboard
├── Churn_Modelling.csv      # dataset (download from Kaggle)
└── README.md
Run the dashboard
Install the requirements:
bash
pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn
Put Churn_Modelling.csv next to the script, then set the path at the top of banking_streamlit.py:
python
DATA_PATH = "Churn_Modelling.csv"
Start the app:
bash
streamlit run banking_streamlit.py
Dashboard pages

Country, Gender, Age, Balance, Products, Active Member, Model. Each page has a chart and a conclusion box, and a collapsible Final Message sits above the KPIs.

Tech stack

Python, pandas, NumPy, Plotly, Matplotlib, Seaborn, scikit-learn, Streamlit.
