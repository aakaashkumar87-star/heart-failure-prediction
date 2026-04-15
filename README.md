# Heart Disease Prediction

## Objective

The objective of this project is to analyze patient health data and identify the key factors that contribute to heart disease. Additionally, a basic machine learning model is built to predict whether a patient is likely to have heart disease.

---

## Dataset

The dataset used in this project is obtained from Kaggle.

**Dataset Link:**
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

**Dataset File:** `heart.csv`

---

## Tools and Technologies Used

* Python 
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* GitHub

---

## Analysis Performed

### 🔹 Data Preprocessing

* Checked for missing values
* Converted categorical variables into numerical format
* Cleaned and structured the dataset

### 🔹 Exploratory Data Analysis (EDA)

* Age distribution of patients
* Gender-wise analysis of heart disease
* Correlation between different health parameters

### 🔹 Data Visualization

* Histogram for age distribution
* Bar chart for gender vs heart disease
* Heatmap for correlation analysis

### 🔹 Machine Learning Model

* Logistic Regression model used
* Dataset split into training and testing sets
* Model trained and evaluated

---

## Results

* The model achieved a good accuracy in predicting heart disease.
* Key factors influencing heart disease include:

  * Age
  * Cholesterol level
  * Maximum heart rate
  * Chest pain type

---

## Screenshots

Include the following screenshots in your project:

* Dataset preview (`df.head()`)
* Visualizations (charts and graphs)
* Correlation heatmap
* Confusion matrix

---

## Project Structure

heart-disease-prediction/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── heart_analysis.ipynb
│
├── screenshots/
│
├── README.md
└── requirements.txt

---

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   ```

2. Navigate to the project folder:

   ```bash
   cd heart-disease-prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open and run `heart_analysis.ipynb`

---

## Conclusion

This project demonstrates how data analysis and machine learning can be used to predict heart disease. The insights gained from the data can help in early detection and better healthcare decision-making.

---

##
