# California Housing Price Prediction

A machine learning project using the California Housing dataset with 20640 instances to predict median house values based on demographic and geographic features.

## Overview

This notebook trains and evaluates different regression models and choosing the model with the highest success rate using the following steps:

1. **Feature Engineering**
   - Created new ratio-based features
   - Added distance-based features from each point to major California cities
   - Scaled area density and added clustering labels

2. **Target Transformation**
   - Applied log transformation to the target (`MedHouseVal`) to reduce skew

3. **Hyperparameter Tuning**
   - Tuned models using `GridSearchCV` and cross-validation
   - Selected the best combination based on validation R² score

4. **Model Training**
   - Used the best setup to retrain on the full training data

5. **Evaluation**
   - R² score (train): ~0.991
   - R² score (test): ~0.874

## Files

- `housing_train.csv`: Training dataset  
- `housing_test.csv`: Test dataset for final evaluation  
- Jupyter Notebook: Contains the full implementation

## How to Run

Open the notebook and run all cells in order. Make sure `housing_train.csv` and `housing_test.csv` are in the same directory.
