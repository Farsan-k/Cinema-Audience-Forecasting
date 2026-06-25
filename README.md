# Cinema Audience Forecasting

## Overview

Cinema Audience Forecasting is an end-to-end Machine Learning project designed to predict the number of people expected to attend a movie based on various influencing factors. Accurate audience forecasting helps cinema operators, production houses, and distributors make informed business decisions related to scheduling, marketing, staffing, and revenue optimization.

This project demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and deployment through a FastAPI application.

---

## Problem Statement

The entertainment industry faces uncertainty in estimating audience turnout for movies. Incorrect demand predictions can result in:

- Poor resource allocation
- Revenue loss
- Underutilized or overcrowded theaters
- Inefficient marketing campaigns

This project aims to develop a predictive model capable of estimating cinema audience attendance using historical data and relevant features.

---

## Solution

A machine learning model is trained using historical cinema data to forecast audience attendance. The system processes raw data, performs feature engineering, evaluates multiple machine learning algorithms, and deploys the best-performing model as a REST API using FastAPI.

---

## Features

- End-to-end machine learning pipeline
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multiple model experimentation
- Model performance evaluation
- FastAPI-based prediction service
- Production-ready project structure
- Serialized model for deployment

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| API Framework | FastAPI |
| Model Serialization | Joblib / Pickle |
| Development | Jupyter Notebook |

---

## Project Structure

```text
Cinema-Audience-Forecasting
│
├── app.py
├── requirements.txt
├── best_model.pkl
│
├── data
│   ├── train.zip
│   └── test.csv
│
├── notebooks
│   └── EDA.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
└── README.md
```

---

## Dataset

The dataset contains historical movie-related information used for audience prediction.

Example features include:

- Movie Genre
- Language
- Release Timing
- Budget
- Marketing Information
- Theater Information
- Historical Audience Trends

Target Variable:

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

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Open the API documentation

```
http://127.0.0.1:8000/docs
```

---

## Machine Learning Workflow

```text
Raw Dataset
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

The project includes comprehensive exploratory analysis to understand:

- Feature distributions
- Missing values
- Correlation between variables
- Outlier detection
- Data quality assessment
- Target variable distribution

---

## Feature Engineering

Data preprocessing includes:

- Missing value handling
- Categorical variable encoding
- Feature transformation
- Data normalization
- Feature selection

---

## Machine Learning Models

Different machine learning algorithms can be evaluated to identify the best-performing forecasting model.

Typical models include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost (if applicable)

The best-performing model is exported for production use.

---

## Model Evaluation

The forecasting model is evaluated using standard regression metrics such as:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help measure prediction accuracy and overall model performance.

---

## API Prediction

Example request

```json
{
    "feature_1": "...",
    "feature_2": "...",
    "feature_3": "..."
}
```

Example response

```json
{
    "predicted_audience": 5423
}
```

---

## Business Applications

This solution can be used by:

- Cinema Chains
- Movie Production Companies
- Film Distributors
- Event Management Companies
- Marketing Teams
- Entertainment Analytics Platforms

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
- Model Deployment
- REST API Development

---

## Future Improvements

- Cloud deployment using AWS or Azure
- Real-time prediction service
- Automated model retraining
- Dashboard for forecasting analytics
- Advanced ensemble learning techniques
- MLOps integration with MLflow
- Docker containerization
- CI/CD pipeline implementation

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Author

**Farsan K**

Aspiring Data Scientist | Machine Learning Engineer

GitHub: https://github.com/Farsan-k

LinkedIn: *Add your LinkedIn Profile*
