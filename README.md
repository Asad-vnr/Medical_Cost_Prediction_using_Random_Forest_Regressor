# Medical_Cost_Prediction_using_Random_Forest_Regressor
Project Overview 🔎

This project focuses on predicting individual medical costs billed by a health insurance company. Using a Random Forest Regressor, a powerful ensemble learning model, we analyze a dataset of patient information to build a predictive model.

The project covers a complete machine learning workflow, including data exploration, feature engineering for categorical data, model training, and a thorough evaluation of the model's performance. It serves as a practical example of how to tackle a real-world regression problem.

Dataset 📊

The project utilizes the "Medical Cost Personal Datasets" from Kaggle, which contains medical insurance information for a set of patients.

    Source: Medical Cost Personal Datasets on Kaggle

    File Used: insurance.csv

The dataset includes the following key columns:

    age: Age of the primary beneficiary.

    sex: Gender of the beneficiary.

    bmi: Body Mass Index.

    children: Number of children covered by health insurance.

    smoker: Whether the person is a smoker or not.

    region: The beneficiary's residential area in the US.

    charges: (This is our target variable) Individual medical costs billed by the insurance.

Project Workflow ⚙️

The project follows a structured, step-by-step methodology:

    Data Loading and Exploration: The insurance.csv dataset is loaded, and an initial Exploratory Data Analysis (EDA) is performed to understand its structure, check for missing values, and review statistical summaries.

    Feature Engineering: All categorical features (sex, smoker, region) are converted into a numerical format using one-hot encoding, making them suitable for the machine learning model.

    Outlier Analysis: The data is visualized using box plots to identify potential outliers. While this step is performed for analysis, the outlier removal process is noted as optional because tree-based models like Random Forest are naturally robust to extreme values.

    Data Visualization: Histograms are plotted for all features to visualize their distributions.

    Data Splitting and Scaling:

        The dataset is split into an 80% training set and a 20% testing set.

        Features are scaled using StandardScaler to ensure all variables are on a comparable scale, which is good practice.

    Model Training: A RandomForestRegressor with 100 trees is trained on the preprocessed training data. The model's Out-of-Bag (OOB) score is checked as a reliable internal measure of its performance.

    Model Evaluation: The model's predictive power is evaluated on the unseen test set using standard regression metrics:

        R-squared: Measures how much of the variance in medical charges the model can explain.

        Mean Absolute Error (MAE): The average absolute difference between the predicted and actual costs.

        Root Mean Squared Error (RMSE): The square root of the average of squared differences, providing an error metric in dollars.

    Visualization of Results:

        A scatter plot of Actual vs. Predicted Charges is created to visually assess the model's accuracy.

        A Feature Importance plot is generated to identify which patient attributes (e.g., smoking status, BMI, age) were most influential in predicting medical costs.

Technologies Used 💻

    Python 3

    Pandas (for data manipulation)

    NumPy (for numerical operations)

    Scikit-learn (for modeling and preprocessing)

    Matplotlib & Seaborn (for data visualization)

    Jupyter Notebook / Google Colab

How to Run This Project ▶️

    Download Files:

        Obtain the project notebook (.ipynb file).

        Download the insurance.csv dataset from the Kaggle link provided above.

    Setup Environment:

        Open the notebook file in Google Colab or a local Jupyter Notebook instance.

        Upload the insurance.csv file to the same environment.

    Execute Code:

        Run the notebook cells sequentially from top to bottom to replicate the entire analysis and model training process.

Results and Conclusion 💡

The Random Forest model demonstrates strong performance in predicting medical costs, as shown by a high R-squared value. The feature importance plot reveals that being a smoker is by far the most significant predictor of higher medical charges, followed by BMI and age. This provides actionable insights into the key drivers of healthcare costs.
