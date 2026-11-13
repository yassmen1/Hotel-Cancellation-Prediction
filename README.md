
# Hotel Booking Cancellation Prediction

## Project Overview
This project aims to predict whether a hotel booking will be canceled
using machine learning techniques.

The goal is to help hotels understand booking cancellation patterns
and support better reservation and resource planning decisions.

## Project Structure

- app.py: Streamlit web application.
- model_functions.py: Custom feature engineering and preprocessing functions.
- hotel_cancellation_model.pkl: Trained machine learning pipeline.
- hotel_bookings.csv: Hotel booking dataset.
- Hotel_Cancellation_Prediction.ipynb: Project analysis, EDA, preprocessing,
  model training, evaluation, and business insights.
- requirements.txt: Required Python packages.

## Machine Learning Workflow

1. Data understanding and exploratory data analysis.
2. Data cleaning and preprocessing.
3. Feature engineering.
4. Baseline model development.
5. Training and comparison of machine learning models.
6. Hyperparameter tuning using cross-validation.
7. Final model evaluation on the test dataset.
8. Model saving and deployment.

## Final Model

The final model is a tuned Random Forest Classifier
integrated into a preprocessing pipeline.

The pipeline includes:
- Missing value imputation.
- Numerical feature scaling.
- Categorical feature encoding.
- Booking feature engineering.

## Model Evaluation

The final model achieved the following test results:

- Accuracy: 84.65%
- Precision (Canceled): 80%
- Recall (Canceled): 59%
- F1-score (Canceled): 0.68
- ROC-AUC: 0.9120

## Web Application

The project includes a Streamlit application for:
- Dashboard overview.
- Hotel booking cancellation prediction.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Hotel_Cancellation_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Cloudpickle

## Notes

The model was evaluated using a held-out test dataset.
Predictions are intended to support business decisions and
should not be interpreted as guaranteed outcomes.
