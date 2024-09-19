# House Price Prediction using California Housing Dataset

This project utilizes **XGBoost** to predict house prices based on the **California Housing Dataset**. The dataset is sourced from `sklearn.datasets.fetch_california_housing`. Various machine learning techniques are applied to build, train, and evaluate the model.

## Dataset

The dataset used is the **California Housing Dataset**, available via `sklearn`. It includes data about house prices and various features such as the number of rooms, income, and population for different blocks in California.

## Installation

To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`

## Project Structure

The project consists of the following steps:

Data Loading: Load the California housing dataset using fetch_california_housing().
Data Preprocessing: Clean the data and perform a train-test split.
Model Training: Use the XGBoost regressor and perform hyperparameter tuning using GridSearchCV.
Model Evaluation: Evaluate the model using metrics such as MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R-squared (R²) score.
Visualization: Plot actual vs predicted house prices to visualize the model's performance.n

## Project Workflow Overview

Import Libraries: The required libraries (pandas, numpy, xgboost, sklearn) are imported.
Load the Dataset: The California Housing Dataset is loaded using fetch_california_housing from sklearn.datasets.
Data Preprocessing: The dataset is split into features (X) and target (y), followed by splitting into training and testing sets.
Model Training:
An XGBoost regressor model is initialized.
Hyperparameters are tuned using GridSearchCV, which searches for the best combination of parameters like the number of estimators, maximum depth, and learning rate.
Model Evaluation:
After training, the model is evaluated using metrics such as:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R-squared (R²)
Visualization: A scatter plot is created to show the actual vs predicted house prices.

## Expected Results

After running the project, you will obtain:

Mean Absolute Error (MAE): The average magnitude of errors in predictions.
Root Mean Squared Error (RMSE): The square root of the average squared differences between actual and predicted values.
R-squared (R²): The proportion of variance in the dependent variable that is predictable from the independent variables.
Additionally, a visualization will be displayed comparing actual vs predicted house prices, providing insight into how well the model performed.
