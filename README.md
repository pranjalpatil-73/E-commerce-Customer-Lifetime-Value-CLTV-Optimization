# E-commerce-Customer-Lifetime-Value-CLTV-Optimization

---

## Project Overview  
This project focuses on analyzing customer data to predict churn and optimize Customer Lifetime Value (CLTV) for an e-commerce business. By identifying high-risk customers and segmenting the customer base, the project provides actionable insights for targeted retention strategies and improved customer engagement.

---

## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Technologies Used](#technologies-used)  
- [Methodology](#methodology)  
- [Key Findings and Results](#key-findings-and-results)  
- [Conclusion and Next Steps](#conclusion-and-next-steps)  
- [Setup and Usage](#setup-and-usage)  

---

## Project Overview  
The main objective is to build a robust system for predicting customer churn and segmenting customers to maximize their lifetime value. The analysis covers:

- Exploratory Data Analysis (EDA) to understand customer behavior patterns  
- Feature engineering to create meaningful variables for modeling  
- Development of classification models to predict customer churn  
- Implementation of clustering techniques for customer segmentation  
- Formulation of data-driven recommendations for CLTV optimization and retention strategies  

---

## Dataset  
The analysis uses the `sales_and_customer_insights.csv` dataset containing 10,000 entries and 15 columns with key customer and sales information. Important columns include:

- Customer_ID  
- Product_ID  
- Transaction_ID  
- Purchase_Frequency (int64)  
- Average_Order_Value (float64)  
- Most_Frequent_Category (object)  
- Time_Between_Purchases (int64)  
- Region (object)  
- Churn_Probability (float64)  
- Lifetime_Value (float64)  
- Launch_Date (object)  
- Peak_Sales_Date (object)  
- Season (object)  
- Preferred_Purchase_Times (object)  
- Retention_Strategy (object)  

---

## Technologies Used  
- Data Manipulation & Analysis: pandas, numpy  
- Data Visualization: matplotlib, seaborn  
- Machine Learning:  
  - sklearn (train_test_split, GridSearchCV, StandardScaler, OneHotEncoder, Pipeline)  
  - RandomForestClassifier, GradientBoostingClassifier, LogisticRegression  
  - Metrics: classification_report, confusion_matrix, roc_auc_score, accuracy_score  
  - imblearn (SMOTE for imbalanced data)  
  - KMeans clustering  
- Other Utilities: datetime, joblib, warnings  

---

## Methodology  
1. Data Loading & Initial Exploration: Load dataset, check data types, preview samples.  
2. Exploratory Data Analysis (EDA):  
   - Visualize numerical feature distributions  
   - Analyze categorical features  
   - Correlation heatmap  
   - Churn probability across categories  
   - Time-based analysis on tenure and churn  
3. Feature Engineering: Create features like Days_Since_Last_Interaction, CLTV_Category, Is_High_Value_Customer, Churn_Risk_Category.  
4. Churn Prediction Model:  
   - Algorithms: Random Forest, Gradient Boosting, Logistic Regression  
   - Handle class imbalance with SMOTE  
   - Hyperparameter tuning via GridSearchCV  
   - Model evaluation using accuracy, precision, recall, F1, ROC AUC, confusion matrix  
5. Customer Segmentation: Use K-Means clustering on RFM metrics and others to identify distinct segments.  
6. CLTV Optimization Strategy: Generate recommendations for retention and lifetime value improvement.  

---

## Key Findings and Results  
- Churn Risk: Approximately 50.3% customers flagged high risk  
- Top Influencing Factors: Lifetime_Value, Days_Since_Last_Interaction, Tenure_Days  
- Retention Strategies: Varied effectiveness across customer segments  

---

## Conclusion and Next Steps  

### Immediate Actions:  
- Deploy churn prediction system in production  
- Launch targeted retention campaigns for high-risk segments  
- Train customer service teams for proactive engagement  

### Long-Term Recommendations:  
- Implement closed-loop feedback for strategy refinement  
- Expand data collection (customer service interactions, reviews)  
- Develop a combined customer health score  

### Future Enhancements:  
- Explore advanced models (deep learning) for better accuracy  
- Integrate real-time data for dynamic prediction and intervention  
- Conduct A/B tests on retention strategies  
- Build dashboards for stakeholder monitoring  

---

## Setup and Usage  

### Prerequisites:  
- Python 3.x  
- pip package manager  

### Install Dependencies:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

Dataset Placement:
Place sales_and_customer_insights.csv in the same directory as the notebook or update the file path accordingly.

Run the Notebook:

jupyter notebook

Open E-commerce Customer Lifetime Value (CLTV) Optimization.ipynb and run all cells sequentially to perform the analysis, train models, and generate insights.
