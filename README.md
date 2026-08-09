# Titanic-Survival-Prediction-Project

A beginner-friendly machine learning project that predicts whether a passenger
survived the Titanic disaster, based on features like age, sex, class, and fare.
## Overview
This project walks through a complete, simple ML workflow: loading data,
cleaning it, engineering features, training a model, and evaluating it.

## Problem Statement
Given information about a Titanic passenger (age, sex, ticket class, fare paid,
family aboard, etc.), predict whether they survived (1) or did not survive (0).
This is a binary classification problem.
## Dataset
- Source: [Titanic dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
## Approach
- Data cleaning: filled missing `Age` with median, missing `Embarked` with
  the most common port, dropped `Cabin` (too many missing values), dropped
  `Name`/`Ticket`/`PassengerId` (not useful for a simple model)
- Feature engineering: created `FamilySize` (siblings/spouses + parents/children + self)
  and `IsAlone`; encoded `Sex` and `Embarked` as numbers
- Model: Logistic Regression (simple, fast, and easy to interpret — a great
  first model for a classification problem)
- Evaluation: accuracy, confusion matrix, precision/recall/F1
## Results
Results will vary slightly depending on the random split, but on the real
Kaggle Titanic dataset, Logistic Regression with this approach typically
achieves ~78–80% accuracy on the test split.
## Tech Stack
Python, Pandas, NumPy, scikit-learn, joblib
## Exploratory Data Analysis

### Survival Distribution

![Survival Distribution](plots/survival_distribution.png)

### Survival by Gender

![Survival by Gender](plots/survival_by_gender.png)

### Survival by Passenger Class

![Survival by Class](plots/survival_by_class.png)
