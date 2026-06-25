# Cinema Audience Forecasting

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-success)
![License](https://img.shields.io/badge/License-MIT-success)

</p>

---

## Overview

Cinema Audience Forecasting is an end-to-end Machine Learning project developed to predict the expected audience attendance for movies based on historical data and various influencing factors. Accurate forecasting enables cinema operators and production companies to make data-driven decisions regarding scheduling, staffing, marketing campaigns, and resource allocation.

This project demonstrates a complete machine learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and deployment through a FastAPI application.

---

## Problem Statement

Predicting movie audience attendance is a challenging task due to multiple factors such as genre, release timing, marketing efforts, seasonal trends, and historical performance. Inaccurate forecasts often lead to poor resource utilization, revenue loss, and inefficient operational planning.

The objective of this project is to build a robust machine learning model capable of accurately forecasting cinema audience attendance using historical data.

---

## Solution

This project leverages supervised machine learning techniques to analyze historical cinema data and predict future audience attendance. The solution includes comprehensive data preprocessing, feature engineering, model comparison, and deployment through a RESTful API, making it suitable for integration with business applications.

---

## Features

- End-to-end Machine Learning pipeline
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Prediction using trained model
- REST API using FastAPI
- Production-ready project structure
- Model serialization for deployment

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| API Development | FastAPI |
| Model Serialization | Joblib |
| Development Environment | Jupyter Notebook |

---

## Project Structure

```text
Cinema-Audience-Forecasting
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│   ├── train.csv
│   └── test.csv
│
├── notebooks
│   └── EDA.ipynb
│
├── models
│   └── best_model.pkl
│
├── src
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
└── outputs
    ├── figures
    └── reports
```

---

## Dataset

The dataset contains historical movie-related information used for audience prediction.

Example features include:

- Movie Genre
- Language
- Budget
- Marketing Spend
- Release Month
- Number of Screens
- Movie Duration
- Ticket Price
- Historical Audience Trends
- Holiday Indicator

**Target Variable**

- Audience Attendance

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Cinema-Audience-Forecasting.git

cd Cinema-Audience-Forecasting
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the FastAPI application

```bash
uvicorn app:app --reload
```

Open the interactive API documentation

```
http://127.0.0.1:8000/docs
```

---

## Machine Learning Workflow

```text
Historical Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Model Serialization
        │
        ▼
FastAPI Deployment
```

---

## Exploratory Data Analysis

The exploratory analysis focuses on understanding the dataset through:

- Missing value analysis
- Data distribution
- Correlation analysis
- Outlier detection
- Feature relationships
- Target variable analysis

The insights obtained during EDA help improve feature engineering and model performance.

---

## Feature Engineering

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical variables
- Feature scaling
- Data transformation
- Feature selection
- Removing redundant features

These steps improve model accuracy and generalization.

---

## Machine Learning Models

The project evaluates multiple regression algorithms to identify the best-performing model.

Algorithms considered include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

The best model is selected based on evaluation metrics and deployed for inference.

---

## Model Evaluation

Model performance is measured using standard regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics ensure reliable forecasting performance on unseen data.

---

## API Example

### Request

```json
{
  "genre": "Action",
  "budget": 5000000,
  "marketing_spend": 800000,
  "release_month": "June"
}
```

### Response

```json
{
  "predicted_audience": 15432
}
```

---

## Business Applications

This solution can be applied in:

- Cinema Chains
- Movie Production Studios
- Film Distribution Companies
- Entertainment Analytics Platforms
- Event Management Organizations
- Marketing Strategy Teams

The forecasting system enables organizations to optimize operational planning, maximize revenue, and improve customer experience.

---

## Skills Demonstrated

- Python Programming
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Regression Modeling
- Model Evaluation
- FastAPI Development
- REST API Development
- Model Deployment

---

## Future Improvements

Potential enhancements include:

- Cloud deployment using AWS, Azure, or Google Cloud
- Docker containerization
- MLflow experiment tracking
- CI/CD pipeline implementation
- Automated model retraining
- Real-time prediction service
- Interactive dashboard using Streamlit or Power BI
- Ensemble learning techniques for improved accuracy

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Author

**Farsan K**

Aspiring Data Scientist | Machine Learning Engineer

**GitHub:** https://github.com/Farsan-k

**LinkedIn:** https://www.linkedin.com/in/your-linkedin-profile/
