# Salary_Prediction_Project
# 💰 Software Engineer Salary Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project is a Machine Learning application that predicts the estimated salary of a software engineer based on various demographic and professional factors. 

The goal is to help developers and job seekers estimate their market value based on real-world data. The application is built using **Python** and deployed as an interactive web app using **Streamlit**.

### 🎥 Demo
![App Screenshot](path/to/your/screenshot.png)

[**🔴 Live Demo Link**](put_your_link_here_if_deployed)

## 📊 The Dataset
The model was trained using the **Stack Overflow Developer Survey [Year]** dataset. 
* **Source:** [Stack Overflow Public Data](https://insights.stackoverflow.com/survey)
* **Data Cleaning:** The raw data was processed to remove outliers, handle missing values, and categorize countries and education levels to improve model accuracy.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
* **Web Framework:** Streamlit (for the UI)
* **IDE:** Jupyter Notebook / VS Code

## 🧠 Model Development
The following steps were taken to build the predictor:
1.  **Data Preprocessing:** Handling null values, encoding categorical variables (Education, Country), and filtering data for "Full-Time" employment.
2.  **Exploratory Data Analysis (EDA):** Visualizing salary distributions across different countries and experience levels.
3.  **Model Selection:** Tested multiple algorithms including:
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor (Selected for best accuracy)
4.  **Evaluation:** The model was evaluated using **GridSearchCV** to minimize the Mean Squared Error (MSE).

## 📂 Project Structure
```text
├── dataset/
│   └── survey_results_public.csv  # Raw data
├── model/
│   └── saved_steps.pkl            # Pickled model & encoders
├── app.py                         # Streamlit application entry point
├── predict_page.py                # Prediction logic
├── explore_page.py                # Data visualization logic
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
